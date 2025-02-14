import moderngl
import torch

from lcmr.grammar import Object
from lcmr.grammar.shapes import Shape2D
from lcmr.renderer.renderer2d.opengl_renderer2d_internals import OpenGlShapeRenderer, OpenGlShapeRendererOptions
from lcmr.utils.elliptic_fourier_descriptors import reconstruct_contour, triangulate_contour
from lcmr.utils.guards import typechecked


@typechecked
class OpenGlFourierRenderer(OpenGlShapeRenderer):
    def __init__(self, options: OpenGlShapeRendererOptions, simplify_threshold: float = 0.001):
        super().__init__(Shape2D.EFD_SHAPE, options)
        self.simplify_threshold = simplify_threshold

    def init_vao(self, objects: Object):
        self.last_length = objects.shape[0]

        verts = reconstruct_contour(objects.efd, n_points=self.n_verts)
        verts = torch.nn.functional.pad(verts, (0, 1), "constant", 1.0)
        verts = (objects.transformation.matrix[:, None, ...] @ verts[..., None]).squeeze(-1)[..., :2]

        # if verts.shape[0] == 1:
        #   we can simplify the contour (remove some points) easily
        #   TODO: do it for batch
        #   mask = simplify_contour(verts)
        #   verts = torch.masked_select(verts, mask).view(1, -1, 2)

        faces = triangulate_contour(verts[None, ...], contour_only=self.contours_only)[0]

        colors = objects.appearance.color.repeat(1, self.n_verts).reshape(-1, self.n_verts, 3)
        confidence = objects.appearance.confidence.expand(-1, self.n_verts).reshape(-1, self.n_verts, 1)

        self.vert_vbo = self.ctx.buffer(verts.cpu().contiguous().numpy())
        self.color_vbo = self.ctx.buffer(colors.cpu().contiguous().numpy())
        self.confidence_vbo = self.ctx.buffer(confidence.cpu().contiguous().numpy())
        self.ibo = self.ctx.buffer(faces.cpu().contiguous().numpy())

        self.vao = self.ctx.vertex_array(
            self.shader,
            [
                self.vert_vbo.bind("in_vert", layout="2f"),
                self.color_vbo.bind("in_color", layout="3f"),
                self.confidence_vbo.bind("in_confidence", layout="1f"),
            ],
            index_buffer=self.ibo,
            mode=self.ctx.LINES if self.contours_only else self.ctx.TRIANGLES,
        )
        self.vao.scope = self.ctx.scope(self.fbo, moderngl.BLEND)

    def init_shader_code(self):
        self.vertex_shader = """
            #version 330

            in vec2 in_vert;
            in vec3 in_color;
            in float in_confidence;
            out vec4 v_color;

            void main() {
                gl_Position = vec4(in_vert * 2 - 1, 0.0, 1.0);
                v_color = vec4(in_color, in_confidence);
            }
        """
        self.geometry_shader = None
        self.fragment_shader = """
            #version 330

            in vec4 v_color;
            out vec4 f_color;

            void main() {
                f_color = v_color;
            }
        """
