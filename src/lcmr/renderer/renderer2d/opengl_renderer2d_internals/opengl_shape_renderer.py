import abc
from moderngl import Context, VertexArray, Framebuffer
from dataclasses import dataclass

from lcmr.grammar import Object
from lcmr.grammar.shapes import Shape2D
from lcmr.utils.guards import typechecked

@typechecked
@dataclass
class OpenGlShapeRendererOptions:
    ctx: Context
    fbo: Framebuffer
    n_verts: int
    contours_only: bool

@typechecked
class OpenGlShapeRenderer(abc.ABC):
    def __init__(self, objectShape: Shape2D, options: OpenGlShapeRendererOptions):
        self.objectShape = objectShape
        
        self.ctx = options.ctx
        self.fbo = options.fbo
        self.n_verts = options.n_verts
        self.contours_only = options.contours_only

        self.vertex_shader: str
        self.geometry_shader: str
        self.fragment_shader: str
        self.vao: VertexArray

        self.init_shader_code()
        self.shader = self.ctx.program(vertex_shader=self.vertex_shader, geometry_shader=self.geometry_shader, fragment_shader=self.fragment_shader)
        self.last_length = -1

    @abc.abstractmethod
    def init_shader_code(self):
        pass

    @abc.abstractmethod
    def init_vao(self, objects: Object):
        pass

    def render(self, objects: Object):
        mask = (objects.objectShape == self.objectShape.value).squeeze(-1)
        if mask.sum() > 0:
            self.init_vao(objects[mask])
            self.vao.render()
