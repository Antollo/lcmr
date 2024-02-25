import torch
from torch.utils.data import Dataset
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

from lcmr.grammar.scene import Scene
from lcmr.renderer.renderer2d import Renderer2D
from lcmr.dataset.dataset_options import DatasetOptions

def renderer_from_options(options: DatasetOptions) -> Renderer2D:
    return options.Renderer(raster_size=options.raster_size, background_color=options.background_color, device=options.renderer_device)

def init_worker(options: DatasetOptions):
    global global_renderer
    global_renderer = renderer_from_options(options)


def regenerate_job(options: DatasetOptions, arg_renderer: Optional[Renderer2D] = None):
    n_objects = options.n_objects

    scenes = [
        Scene.from_tensors_sparse(
            torch.rand(1, 1, n_objects, 2),
            torch.rand(1, 1, n_objects, 2) / 5 + 0.05,
            torch.rand(1, 1, n_objects, 1),
            torch.rand(1, 1, n_objects, 3),
            torch.rand(1, 1, n_objects, 1) / 4 + 0.75,
        )
        for _ in range(options.n_samples)
    ]
    scenes = scenes
    images = [(arg_renderer or global_renderer).render(scene)[..., :3].cpu() for scene in scenes]
    return list(zip(images, scenes))

class RandomDataset(Dataset):
    def __init__(self, options: DatasetOptions):
        self.options = options
        self.renderer = None

        if options.n_jobs > 1:
            self.pool = ProcessPoolExecutor(max_workers=options.n_jobs, initargs=(options,), initializer=init_worker)
            self.futures = []
            for _ in range(options.n_jobs):
                self.append_new_job()
        else:
            self.pool = None
        self.regenerate()

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

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
            self.data = regenerate_job(self.options, arg_renderer=self.renderer)
    
    @staticmethod
    def collate_fn(batch):
        if type(batch[0]) is tuple:
            batch = [list(x) for x in zip(*batch)]
            return [torch.cat(x, dim=0).pin_memory() for x in batch]
        else:
            return torch.cat(batch, dim=0).pin_memory()
