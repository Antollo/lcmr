from typing import Optional

import moderngl
import torch
from torchtyping import TensorType

from lcmr.grammar import Layer, Scene
from lcmr.grammar.scene_data import SceneData
from lcmr.renderer.renderer2d.opengl_renderer2d_internals import OpenGlDiskRenderer, OpenGlFourierRenderer, OpenGlShapeRendererOptions
from lcmr.renderer.renderer2d.renderer2d import Renderer2D, height_dim, width_dim
from lcmr.utils.colors import colors
from lcmr.utils.guards import ImageHWC4, typechecked


@typechecked
class OpenGLRenderer2D(Renderer2D):
    def __init__(
        self,
        raster_size: tuple[int, int],
        samples: int = 4,
        background_color: Optional[TensorType[4, torch.float32]] = None,
        n_verts: int = 64,
        device: torch.device = torch.device("cpu"),
        wireframe: bool = False,
        contours_only: bool = False,
    ):
        super().__init__(raster_size=raster_size, device=device)

        if background_color == None:
            background_color = colors.black
        try:
            self.ctx = moderngl.create_standalone_context()
        except:
            # https://github.com/moderngl/moderngl/issues/392
            self.ctx = moderngl.create_standalone_context(backend="egl")
            samples = 1

        self.ctx.gc_mode = "context_gc"
        self.fbo1 = self.ctx.framebuffer([self.ctx.renderbuffer(raster_size[::-1], components=4, samples=samples, dtype="f4")])
        self.fbo2 = self.ctx.framebuffer([self.ctx.renderbuffer(raster_size[::-1], components=4, dtype="f4")])
        # self.buf = self.ctx.buffer(reserve=raster_size[0] * raster_size[1] * 4 * 4)

        self.background_color = background_color.to(device)
        self.background_color_list = background_color.tolist()
        self.background = background_color[None, None, ...].to(device).expand(*raster_size, -1)
        self.ctx.wireframe = wireframe

        self.contours_only = contours_only
        self.n_verts = n_verts

    @property
    def n_verts(self):
        return self._n_verts

    @n_verts.setter
    def n_verts(self, value):
        self._n_verts = value
        options = OpenGlShapeRendererOptions(ctx=self.ctx, fbo=self.fbo1, n_verts=self._n_verts, contours_only=self.contours_only)
        self.shape_renderers = [OpenGlDiskRenderer(options), OpenGlFourierRenderer(options)]

    def __del__(self):
        self.ctx.release()

    def render(self, scene: Scene) -> SceneData:
        self.ctx.gc()

        if len(scene) == 1:
            imgs = self.render_scene(scene)
        else:
            imgs = []
            # somehow iterating scene like that is a little bit time consuming
            # (maybe it's creating copies?)
            for single_scene in scene:
                img = self.render_scene(single_scene)
                imgs.append(img)
            imgs = torch.cat(imgs, dim=0)

        return SceneData(
            scene=scene,
            image=imgs,
            batch_size=[len(scene)],
        )

    def render_layer(self, layer: Layer) -> ImageHWC4:

        # Online tool to visualize OpenGL blenfing on examples https://www.andersriggelsen.dk/glblendfunc.php
        self.ctx.blend_func = self.ctx.SRC_ALPHA, self.ctx.ONE, self.ctx.ONE, self.ctx.ONE
        self.ctx.blend_equation = self.ctx.FUNC_ADD

        # self.fbo1.clear(*self.background_color_list[0:3], 0)
        self.fbo1.clear(0, 0, 0, 0)
        for shape_renderer in self.shape_renderers:
            shape_renderer.render(layer.object)
        self.ctx.copy_framebuffer(self.fbo2, self.fbo1)

        # Using opengl-cuda interop is actually slower on small images,
        # on my machine it's faster on images of size 1000x1000 and larger.
        # + It's not supported on WSL (but it is supported on windows and linux).

        # from cuda import cudart
        # import cupy as cp
        # self.fbo2.read_into(self.buf, components=4, dtype="f4")
        # result, resource = cudart.cudaGraphicsGLRegisterBuffer(self.buf.glo, cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsReadOnly)
        # assert result == 0
        # result = cudart.cudaGraphicsMapResources(1, resource, cudart.cudaGraphicsMapFlags.cudaGraphicsMapFlagsReadOnly)[0]
        # assert result == 0
        # result, ptr, size = cudart.cudaGraphicsResourceGetMappedPointer(resource)
        # assert result == 0
        # cuda_buffer = cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(ptr, size, owner=None), 0)
        # tensor = torch.from_dlpack(cp.ndarray(
        #    shape=(size//4,),
        #    dtype=cp.float32,
        #    strides=None,
        #    order='C',
        #    memptr=cuda_buffer,
        # )).reshape((*self.raster_size, 4)).to(self.device).clone()
        # cudart.cudaGraphicsUnmapResources(1, resource, 0)[0]
        # assert result == 0
        # cudart.cudaGraphicsUnregisterResource(resource)[0]
        # assert result == 0

        rendered_layer = torch.frombuffer(bytearray(self.fbo2.read(components=4, dtype="f4")), dtype=torch.float32).reshape((*self.raster_size, 4)).to(self.device)

        alpha = rendered_layer[..., 3, None]
        rgb = rendered_layer[..., :3]
        rgb /= alpha.clamp(min=0.0001)
        rendered_layer.clamp_(0, 1)

        return rendered_layer

    def render_scene(self, single_scene: Scene) -> TensorType[1, height_dim, width_dim, 4, torch.float32]:
        img = self.background

        if len(single_scene.layer) == 1:
            rendered_layer = self.render_layer(single_scene.layer)
            img = self.alpha_compositing(rendered_layer, img)
        else:
            for layer in single_scene.layer:
                rendered_layer = self.render_layer(layer)
                img = self.alpha_compositing(rendered_layer, img)

        return img[None, ...]
