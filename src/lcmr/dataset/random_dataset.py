import platform
from concurrent.futures import ProcessPoolExecutor
from math import ceil
from typing import Optional

import numpy as np
import torch
from scipy import spatial
from tqdm import tqdm

from lcmr.dataset.any_dataset import AnyDataset
from lcmr.dataset.dataset_options import DatasetOptions
from lcmr.grammar.scene import Scene
from lcmr.grammar.scene_data import SceneData
from lcmr.grammar.shapes import Shape2D
from lcmr.renderer.renderer2d import Renderer2D
from lcmr.utils.elliptic_fourier_descriptors import EfdGenerator, compute_rotational_symmetry

if platform.system() == "Linux":
    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
        print("Setting start method to 'spawn' failed")


def random_translation_inner(n: int) -> np.ndarray:
    candidates = []
    while len(candidates) < n:
        points = np.random.rand(n * 2, 2).astype(np.float32)
        candidates = points[spatial.ConvexHull(points).vertices]
    indices = np.random.choice(candidates.shape[0], size=n, replace=False)
    return candidates[indices]


def random_translation(b: int, n: int) -> np.ndarray:
    return torch.from_numpy(np.array([random_translation_inner(n) for _ in range(b)]))


def min_max_dist_translation(batch_size: int, n_objects: int, n_tries: int = 8):
    translation = torch.rand(batch_size, 1, n_tries, n_objects, 2)
    dist = torch.cdist(translation, translation) + torch.zeros((n_objects, n_objects)).fill_diagonal_(torch.inf)[None, None, None]
    idx = dist.flatten(-2, -1).amin(dim=-1).argmax(dim=-1)
    translation = torch.gather(translation, 2, idx[..., None, None, None].expand(-1, -1, -1, *translation.shape[3:])).squeeze(2)
    return translation


def renderer_from_options(options: DatasetOptions) -> Renderer2D:
    return options.Renderer(raster_size=options.raster_size, background_color=options.background_color, device=options.renderer_device)


def to_unary(indices: torch.Tensor, n: int) -> torch.Tensor:
    tril_matrix = torch.ones((n, n), device=indices.device).tril_()
    return tril_matrix[indices]


global_renderer: Optional[Renderer2D] = None
global_fdg: Optional[EfdGenerator] = None


def init_worker(options: DatasetOptions):
    global global_renderer

    global_renderer = renderer_from_options(options)
    if options.efd_options != None:
        global global_fdg
        global_fdg = EfdGenerator(options.efd_options)


def regenerate_job(options: DatasetOptions, arg_renderer: Optional[Renderer2D] = None, arg_fdg: Optional[EfdGenerator] = None) -> SceneData:
    n_objects = options.n_objects
    renderer = arg_renderer or global_renderer
    fdg = arg_fdg or global_fdg

    batch_size = options.renderer_batch_size

    with torch.no_grad():
        scenes = []
        for _ in range(ceil(options.n_samples / batch_size)):
            n_visible = torch.randint(0, n_objects, (batch_size, 1))
            efd = None if fdg == None else fdg.sample(batch_size * n_objects).view(batch_size, 1, n_objects, -1, 4)
            rotation_vec = torch.nn.functional.normalize(torch.rand(batch_size * n_objects, 2) * 2 - 1, dim=-1).view(batch_size, 1, n_objects, 2)

            # if efd != None:
            #     sym = compute_rotational_symmetry(efd)
            #     rotation_vec[sym==2, 0] = rotation_vec[sym==2, 0].abs()
            #     rotation_vec[sym==4] = rotation_vec[sym==4].abs()
            #
            #     efd = efd.view(batch_size, 1, n_objects, -1, 4)
            # rotation_vec = rotation_vec.view(batch_size, 1, n_objects, 2)

            scene = Scene.from_tensors_sparse(
                translation=min_max_dist_translation(batch_size, n_objects) * 0.9 + 0.05,
                scale=(torch.rand(batch_size, 1, n_objects, 1).expand(-1, -1, -1, 2) if options.use_single_scale else torch.rand(batch_size, 1, n_objects, 2)) * 0.2 + 0.1,
                # angle=torch.rand(batch_size, 1, n_objects, 1) * torch.pi * 2,
                rotation_vec=rotation_vec,
                color=torch.rand(batch_size, 1, n_objects, 3),
                # confidence=to_unary(n_visible, n_objects)[..., None] - torch.rand(batch_size, 1, n_objects, 1) * 0.1,
                confidence=torch.ones(batch_size, 1, n_objects, 1) - torch.rand(batch_size, 1, n_objects, 1) * 0.1,
                objectShape=torch.ones(batch_size, 1, n_objects, 1, dtype=torch.uint8) * (Shape2D.DISK.value if fdg == None else Shape2D.EFD_SHAPE.value),
                efd=efd,
                backgroundColor=torch.rand(batch_size, 3) if options.background_color == None else None,
                device=renderer.device,
            )
            scenes.append(scene)

        if options.n_samples % batch_size != 0:
            scenes[-1] = scenes[-1][: (options.n_samples % batch_size)]

        scene_iterator = tqdm(scenes, desc="[RandomDataset] Rendering scenes") if options.verbose else scenes

        images = [renderer.render(scene).image[..., :3].cpu() for scene in scene_iterator] if options.return_images else None
        scenes = [scene.cpu() for scene in scenes]

        data = []
        for image, scene in zip(images, scenes):
            b = len(scene) if options.return_scenes else len(image) if options.return_images else None
            data.append(
                SceneData(
                    scene=scene if options.return_scenes else None,
                    image=image if options.return_images else None,
                    batch_size=[b],
                    device=torch.device("cpu"),
                )
            )

    return data


class RandomDataset(AnyDataset):
    def __init__(self, options: DatasetOptions):
        super().__init__(None)

        self.options = options
        self.renderer: Optional[Renderer2D] = None
        self.fdg: Optional[EfdGenerator] = None
        self.len: int = None

        if options.n_jobs > 1:
            self.pool = ProcessPoolExecutor(max_workers=options.n_jobs, initargs=(options,), initializer=init_worker, mp_context=torch.multiprocessing)
            self.futures = []
            for _ in range(options.n_jobs):
                self.append_new_job()
        else:
            self.pool = None

        self.regenerate()

    def __getitem__(self, idx: int):
        batch_size = self.options.renderer_batch_size
        data = self.data[idx // batch_size]
        i = idx % batch_size
        if type(data) is tuple:
            return tuple(x[i, None] for x in data)
        else:
            return data[i, None]

    def __len__(self) -> int:
        if self.len == None:
            self.len = sum((len(d) for d in self.data))
        return self.len

    def append_new_job(self):
        self.futures.append(self.pool.submit(regenerate_job, self.options))

    def regenerate(self):
        if self.pool != None:
            self.append_new_job()
            self.data = self.futures[0].result()
            del self.futures[0]
        else:
            if self.renderer == None:
                self.renderer = renderer_from_options(self.options)
                if self.options.efd_options != None:
                    self.fdg = EfdGenerator(self.options.efd_options)
            self.data = regenerate_job(self.options, arg_renderer=self.renderer, arg_fdg=self.fdg)
