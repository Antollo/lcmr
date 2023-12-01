import torch
from torchtyping import TensorType
import moderngl
from math import pi

from .renderer2d import Renderer2D, raster_dim
from ...grammar import Scene, Layer
from ...grammar.guards import typechecked, batch_dim


@typechecked
class OpenGLRenderer2D(Renderer2D):
    def __init__(self, raster_size: tuple[int, int], samples: int = 4):
        super().__init__(raster_size)

        try:
            self.ctx = moderngl.create_standalone_context()
            self.fbo1 = self.ctx.framebuffer([self.ctx.renderbuffer(raster_size[::-1], components=samples, samples=samples)])
        except:
            # https://github.com/moderngl/moderngl/issues/392
            self.ctx = moderngl.create_standalone_context(backend="egl")
            self.fbo1 = self.ctx.framebuffer([self.ctx.renderbuffer(raster_size[::-1], components=4, samples=samples)])

        self.fbo2 = self.ctx.framebuffer([self.ctx.renderbuffer(raster_size[::-1], components=4)])
        self.init_shader_code()
        self.shader = self.ctx.program(vertex_shader=self.vertex_shader, geometry_shader=self.geometry_shader, fragment_shader=self.fragment_shader)

        self.last_length = -1

    def render(self, scene: Scene) -> TensorType[batch_dim, raster_dim, raster_dim, 4, torch.float32]:
        imgs = []
        for single_scene in scene:
            img = torch.zeros((*self.raster_size, 4))

            # composition is ignored
            for layer in single_scene.layer:
                rendered_layer = self.render_layer(layer)
                img = self.alpha_compositing(rendered_layer, img)

            imgs.append(img[None, ...])

        return torch.vstack(imgs)

    def init_vao(self, layer: Layer):
        if self.last_length != layer.object.shape[0]:
            self.last_length = layer.object.shape[0]

            # only accept transformation as 3x3 matrix
            self.mat_vbo = self.ctx.buffer(layer.object.transformation.matrix.reshape(-1).numpy(), dynamic=True)
            self.color_vbo = self.ctx.buffer(layer.object.appearance.color.reshape(-1).numpy(), dynamic=True)
            self.confidence_vbo = self.ctx.buffer(layer.object.appearance.confidence.reshape(-1).numpy(), dynamic=True)

            self.vao = self.ctx.vertex_array(
                self.shader,
                [
                    self.mat_vbo.bind("in_mat", layout="9f"),
                    self.color_vbo.bind("in_color", layout="3f"),
                    self.confidence_vbo.bind("in_confidence", layout="1f"),
                ],
                mode=self.ctx.POINTS,
            )
            self.vao.scope = self.ctx.scope(self.fbo1, moderngl.BLEND)
        else:
            self.mat_vbo.write(layer.object.transformation.matrix.reshape(-1).numpy())
            self.color_vbo.write(layer.object.appearance.color.reshape(-1).numpy())
            self.confidence_vbo.write(layer.object.appearance.confidence.reshape(-1).numpy())

    def render_layer(self, layer: Layer) -> TensorType[raster_dim, raster_dim, 4, torch.float32]:
        self.init_vao(layer)

        self.fbo1.clear(0, 0, 0, 0)
        self.vao.render()
        self.ctx.copy_framebuffer(self.fbo2, self.fbo1)

        return torch.frombuffer(bytearray(self.fbo2.read(components=4, dtype="f4")), dtype=torch.float32).reshape((*self.raster_size, 4))

    def alpha_compositing(
        self, src: TensorType[raster_dim, raster_dim, 4, torch.float32], dst: TensorType[raster_dim, raster_dim, 4, torch.float32]
    ) -> TensorType[raster_dim, raster_dim, 4, torch.float32]:
        # https://stackoverflow.com/a/60401248/14344875

        src_alpha = src[..., 3, None]
        dst_alpha = dst[..., 3, None]
        out_alpha = src_alpha + dst_alpha * (1 - src_alpha)

        src_rgb = src[..., :3]
        dst_rgb = dst[..., :3]
        out_rgb = (src_rgb * src_alpha + dst_rgb * dst_alpha * (1 - src_alpha)) / torch.clamp(out_alpha, min=0.0001)

        return torch.dstack((out_rgb, out_alpha))

    def init_shader_code(self):
        self.vertex_shader = """
            #version 330

            in mat3 in_mat;
            in vec3 in_color;
            in float in_confidence;
            out vec4 v_color;
            out mat3 v_mat;

            void main() {
                gl_Position = vec4(0, 0, 0, 1);
                v_color = vec4(in_color, in_confidence);
                v_mat = in_mat;
            }
        """
        v_count = min(63, self.ctx.info["GL_MAX_GEOMETRY_OUTPUT_COMPONENTS"] // 2 + 1)
        self.geometry_shader = f"""
            #version 330
            
            const int v_count = {v_count};
            const float pi = {pi};
            const float radius = 1;
            const float angle = 2 * pi / v_count;
            
            layout (points) in;
            in vec4 v_color[];
            in mat3 v_mat[];
            layout (triangle_strip, max_vertices = {v_count * 2 + 1}) out;
            out vec4 g_color;
            
            void main() {{
                for (int i = 0; i <= v_count; i++)
                {{
                    float currentAngle = angle * i;
                    float x = radius * cos(currentAngle);
                    float y = radius * sin(currentAngle);
                    
                    vec2 position = (vec3(x, y, 1) * v_mat[0]).xy;
                    position.y = 1 - position.y;
                    gl_Position = vec4(position * 2 - 1, 0, 1);
                    g_color = v_color[0];
                    EmitVertex();
                    
                    position = (vec3(0, 0, 1) * v_mat[0]).xy;
                    position.y = 1 - position.y;
                    gl_Position = vec4(position * 2 - 1, 0, 1);
                    g_color = v_color[0];
                    EmitVertex();
                }}
                EndPrimitive();
            }}
        """
        self.fragment_shader = """
            #version 330

            in vec4 g_color;
            out vec4 f_color;

            void main() {
                f_color = g_color;
            }
        """
