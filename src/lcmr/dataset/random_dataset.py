import torch
from concurrent.futures import ProcessPoolExecutor
from typing import Optional
import platform

from lcmr.grammar.scene import Scene
from lcmr.grammar.shapes import Shape2D
from lcmr.renderer.renderer2d import Renderer2D
from lcmr.utils.fourier_shape_descriptors import FourierDescriptorsGenerator
from lcmr.dataset.dataset_options import DatasetOptions
from lcmr.dataset.any_dataset import AnyDataset

if platform.system() == "Linux":
    try:
        torch.multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass


def renderer_from_options(options: DatasetOptions) -> Renderer2D:
    return options.Renderer(raster_size=options.raster_size, background_color=options.background_color, device=options.renderer_device)


global_renderer: Optional[Renderer2D] = None
global_fdg: Optional[FourierDescriptorsGenerator] = None


def init_worker(options: DatasetOptions):
    global global_renderer

    global_renderer = renderer_from_options(options)
    if options.fourier_shapes_options != None:
        global global_fdg
        global_fdg = FourierDescriptorsGenerator(options.fourier_shapes_options)


def regenerate_job(options: DatasetOptions, arg_renderer: Optional[Renderer2D] = None, arg_fdg: Optional[FourierDescriptorsGenerator] = None):
    n_objects = options.n_objects
    renderer = arg_renderer or global_renderer
    fdg = arg_fdg or global_fdg

    with torch.no_grad():
        scenes = [
            Scene.from_tensors_sparse(
                translation=torch.rand(1, 1, n_objects, 2),
                scale=torch.rand(1, 1, n_objects, 2) / 5 + 0.05,
                angle=torch.rand(1, 1, n_objects, 1),
                color=torch.rand(1, 1, n_objects, 3),
                confidence=torch.ones(1, 1, n_objects, 1),
                objectShape=torch.ones(1, 1, n_objects, 1, dtype=torch.uint8) * (Shape2D.DISK.value if fdg == None else Shape2D.FOURIER_SHAPE.value),
                fourierCoefficients=None if fdg == None else fdg.sample(n_objects)[None, None, ...],
            )
            for _ in range(options.n_samples)
        ]
        images = [renderer.render(scene)[..., :3].cpu() for scene in scenes] if options.return_images else None

    if options.return_scenes and options.return_images:
        return list(zip(images, scenes))
    if options.return_scenes:
        return scenes
    if options.return_images:
        return images
    # TODO: warning?
    return tuple()

class RandomDataset(AnyDataset):
    def __init__(self, options: DatasetOptions):
        super().__init__(None)
        
        self.options = options
        self.renderer: Optional[Renderer2D] = None
        self.fdg: Optional[FourierDescriptorsGenerator] = None

        if options.n_jobs > 1:
            self.pool = ProcessPoolExecutor(max_workers=options.n_jobs, initargs=(options,), initializer=init_worker, mp_context=torch.multiprocessing)
            self.futures = []
            for _ in range(options.n_jobs):
                self.append_new_job()
        else:
            self.pool = None
        self.regenerate()

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
                if self.options.fourier_shapes_options != None:
                    self.fdg = FourierDescriptorsGenerator(self.options.fourier_shapes_options)
            self.data = regenerate_job(self.options, arg_renderer=self.renderer, arg_fdg=self.fdg)