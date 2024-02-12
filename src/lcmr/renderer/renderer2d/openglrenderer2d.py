import torch
from torchtyping import TensorType
import moderngl
from math import pi

from lcmr.renderer.renderer2d.renderer2d import Renderer2D, height_dim, width_dim
from lcmr.grammar import Scene, Layer
from lcmr.utils.guards import typechecked, ImageBHWC4, ImageHWC4


@typechecked
class OpenGLRenderer2D(Renderer2D):
    def __init__(
        self,
        raster_size: tuple[int, int],
        samples: int = 4,
        background_color: TensorType[4, torch.float32] = torch.zeros(4),
        gamma_rgb: float = 1.0,
        gamma_confidence: float = 1.0,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(raster_size)

        try:
            self.ctx = moderngl.create_standalone_context()
            self.fbo1 = self.ctx.framebuffer([self.ctx.renderbuffer(raster_size[::-1], components=4, samples=samples, dtype="f4")])
        except:
            # https://github.com/moderngl/moderngl/issues/392
            self.ctx = moderngl.create_standalone_context(backend="egl")
            self.fbo1 = self.ctx.framebuffer([self.ctx.renderbuffer(raster_size[::-1], components=4, samples=1, dtype="f4")])

        self.fbo2 = self.ctx.framebuffer([self.ctx.renderbuffer(raster_size[::-1], components=4, dtype="f4")])
        # self.buf = self.ctx.buffer(reserve=raster_size[0] * raster_size[1] * 4 * 4)
        self.init_shader_code(gamma_rgb, gamma_confidence)
        self.shader = self.ctx.program(vertex_shader=self.vertex_shader, geometry_shader=self.geometry_shader, fragment_shader=self.fragment_shader)
        self.device = device
        self.background_color = background_color.to(device)
        self.background_color_list = background_color.tolist()
        self.background = background_color[None, None, ...].to(device).repeat(*raster_size, 1)

        self.last_length = -1

    def render(self, scene: Scene) -> ImageBHWC4:
        if len(scene) == 1:
            return self.render_scene(scene)
        else:
            imgs = []
            # somehow iterating scene like that is a little bit time consuming
            # (maybe it's creating copies?)
            for single_scene in scene:
                img = self.render_scene(single_scene)
                imgs.append(img)
            return torch.cat(imgs, dim=0)

    def init_vao(self, layer: Layer):
        if self.last_length != layer.object.shape[0]:
            self.last_length = layer.object.shape[0]

            # only accept transformation as 3x3 matrix
            self.mat_vbo = self.ctx.buffer(layer.object.transformation.matrix.reshape(-1).detach().cpu().numpy(), dynamic=True)
            self.color_vbo = self.ctx.buffer(layer.object.appearance.color.reshape(-1).detach().cpu().numpy(), dynamic=True)
            self.confidence_vbo = self.ctx.buffer(layer.object.appearance.confidence.reshape(-1).detach().cpu().numpy(), dynamic=True)

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
            self.mat_vbo.write(layer.object.transformation.matrix.reshape(-1).detach().cpu().numpy())
            self.color_vbo.write(layer.object.appearance.color.reshape(-1).detach().cpu().numpy())
            self.confidence_vbo.write(layer.object.appearance.confidence.reshape(-1).detach().cpu().numpy())

    def render_layer(self, layer: Layer) -> ImageHWC4:
        self.init_vao(layer)

        # Online tool to visualize OpenGL blenfing on examples https://www.andersriggelsen.dk/glblendfunc.php
        self.ctx.blend_func = self.ctx.SRC_ALPHA, self.ctx.ONE, self.ctx.ONE, self.ctx.ONE
        self.ctx.blend_equation = self.ctx.FUNC_ADD

        #self.fbo1.clear(*self.background_color_list[0:3], 0)
        self.fbo1.clear(0, 0, 0, 0)
        self.vao.render()
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

        rendered_layer = (
            torch.frombuffer(bytearray(self.fbo2.read(components=4, dtype="f4")), dtype=torch.float32).reshape((*self.raster_size, 4)).to(self.device)
        )

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

    def init_shader_code(self, gamma_rgb: float, gamma_confidence: float):
        self.vertex_shader = f"""
            #version 330
            
            const float gamma_rgb = {gamma_rgb};
            const float gamma_confidence = {gamma_confidence};
            const vec4 gamma = vec4(gamma_rgb, gamma_rgb, gamma_rgb, gamma_confidence);

            in mat3 in_mat;
            in vec3 in_color;
            in float in_confidence;
            out vec4 v_color;
            out mat3 v_mat;

            void main() {{
                gl_Position = vec4(0, 0, 0, 1);
                v_color = pow(vec4(in_color, in_confidence), 1 / gamma);
                v_mat = in_mat;
            }}
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
                    gl_Position = vec4(position * 2 - 1, 0, 1);
                    g_color = v_color[0];
                    EmitVertex();
                    
                    position = (vec3(0, 0, 1) * v_mat[0]).xy;
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
