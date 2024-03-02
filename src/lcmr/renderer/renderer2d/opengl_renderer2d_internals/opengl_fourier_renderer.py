import torch
import numpy as np
import moderngl
from moderngl import Context, Framebuffer
from torchtyping import TensorType
from functools import cache
from collections.abc import Iterable
from mapbox_earcut import triangulate_float32

from lcmr.grammar import Object
from lcmr.grammar.shapes import Shape2D
from lcmr.utils.guards import typechecked, object_dim, vec_dim, optional_dims
from lcmr.renderer.renderer2d.opengl_renderer2d_internals import OpenGlShapeRenderer


@cache
@typechecked
def order_phases(n_orders: int, n_points: int, device: torch.device):
    t = torch.linspace(0, 1.0, n_points, device=device)[None, ...]
    orders = torch.arange(1, n_orders, device=device)[..., None]
    order_phases = 2 * np.pi * orders * t
    order_phases = order_phases[None, ...]

    return torch.cos(order_phases), torch.sin(order_phases)


@typechecked
def reconstruct_contour(descriptors: TensorType[optional_dims:..., -1, 4, torch.float32], n_points=64) -> TensorType[optional_dims:..., -1, 2, torch.float32]:
    # based on pyefd.reconstruct_contour

    device = descriptors.device
    descriptors = descriptors[..., None]

    order_phases_cos, order_phases_sin = order_phases(descriptors.shape[-3] + 1, n_points, device)

    xt_all = descriptors[..., 0, :] * order_phases_cos + descriptors[..., 1, :] * order_phases_sin
    yt_all = descriptors[..., 2, :] * order_phases_cos + descriptors[..., 3, :] * order_phases_sin

    xt_all = xt_all.sum(axis=-2)
    yt_all = yt_all.sum(axis=-2)

    reconstruction = torch.stack((xt_all, yt_all), axis=-1)
    return reconstruction


@typechecked
def simplify_contour(contour: TensorType[object_dim, vec_dim, 2, torch.float32], threshold: float = 0.001) -> TensorType[object_dim, vec_dim, 1, torch.bool]:
    contour_padded = torch.nn.functional.pad(contour, (0, 0, 1, 1), "circular")

    a, b, c = contour_padded[:, :-2, :], contour_padded[:, 1:-1, :], contour_padded[:, 2:, :]
    ba = a - b
    bc = c - b
    cosine_angle = torch.bmm(ba.view(-1, 1, 2), bc.view(-1, 2, 1)).view(contour.shape[0], contour.shape[1]) / (
        torch.linalg.norm(ba, dim=-1) * torch.linalg.norm(bc, dim=-1)
    )
    mask = cosine_angle[..., None] > -1 + threshold

    return mask


@typechecked
def triangularize_contour(contours: Iterable[TensorType[-1, 2, torch.float32]] | TensorType[-1, -1, 2, torch.float32]) -> np.ndarray:
    faces_list = []
    total = 0
    for contour in contours:
        faces = triangulate_float32(contour.detach().cpu().numpy(), np.array([len(contour)])).astype(np.int32)
        faces_list.append(faces + total)
        total += contour.shape[0]

    return np.concatenate(faces_list).reshape(-1, 3)


@typechecked
class OpenGlFourierRenderer(OpenGlShapeRenderer):
    def __init__(self, ctx: Context, fbo: Framebuffer, n_verts: int, simplify_threshold: float = 0.001):
        super().__init__(ctx, fbo, Shape2D.FOURIER_SHAPE, n_verts)
        self.simplify_threshold = simplify_threshold

    def init_vao(self, objects: Object):
        self.last_length = objects.shape[0]

        verts = reconstruct_contour(objects.fourierCoefficients, n_points=self.n_verts)
        verts = torch.nn.functional.pad(verts, (0, 1), "constant", 1.0)
        verts = (objects.transformation.matrix[:, None, ...] @ verts[..., None]).squeeze(-1)[..., :2]
        mask = simplify_contour(verts)

        verts = torch.masked_select(verts, mask).view(-1, 2)
        verts = verts.split(mask.sum(axis=-2).flatten().tolist())
        faces = triangularize_contour(verts)

        colors = objects.appearance.color.repeat(1, self.n_verts).reshape(-1, self.n_verts, 3)
        colors = torch.masked_select(colors, mask).view(-1, 3)

        confidence = objects.appearance.confidence.repeat(1, self.n_verts).reshape(-1, self.n_verts, 1)
        confidence = torch.masked_select(confidence, mask).view(-1, 1)

        self.vert_vbo = self.ctx.buffer(torch.cat(verts).detach().cpu().contiguous().numpy())
        self.color_vbo = self.ctx.buffer(colors.cpu().contiguous().numpy())
        self.confidence_vbo = self.ctx.buffer(confidence.cpu().contiguous().numpy())
        self.ibo = self.ctx.buffer(faces)

        self.vao = self.ctx.vertex_array(
            self.shader,
            [
                self.vert_vbo.bind("in_vert", layout="2f"),
                self.color_vbo.bind("in_color", layout="3f"),
                self.confidence_vbo.bind("in_confidence", layout="1f"),
            ],
            index_buffer=self.ibo,
            mode=self.ctx.TRIANGLES,
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
