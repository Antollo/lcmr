import moderngl
from math import pi

from lcmr.grammar import Object
from lcmr.grammar.shapes import Shape2D
from lcmr.utils.guards import typechecked
from lcmr.renderer.renderer2d.opengl_renderer2d_internals import OpenGlShapeRenderer, OpenGlShapeRendererOptions


@typechecked
class OpenGlDiskRenderer(OpenGlShapeRenderer):
    def __init__(self, options: OpenGlShapeRendererOptions):
        super().__init__(Shape2D.DISK, options)

    def init_vao(self, objects: Object):
        if self.last_length != objects.shape[0]:
            self.last_length = objects.shape[0]

            # only accept transformation as 3x3 matrix
            self.mat_vbo = self.ctx.buffer(objects.transformation.matrix.reshape(-1).detach().cpu().numpy(), dynamic=True)
            self.color_vbo = self.ctx.buffer(objects.appearance.color.reshape(-1).detach().cpu().numpy(), dynamic=True)
            self.confidence_vbo = self.ctx.buffer(objects.appearance.confidence.reshape(-1).detach().cpu().numpy(), dynamic=True)

            self.vao = self.ctx.vertex_array(
                self.shader,
                [
                    self.mat_vbo.bind("in_mat", layout="9f"),
                    self.color_vbo.bind("in_color", layout="3f"),
                    self.confidence_vbo.bind("in_confidence", layout="1f"),
                ],
                mode=self.ctx.POINTS,
            )
            self.vao.scope = self.ctx.scope(self.fbo, moderngl.BLEND)
        else:
            self.mat_vbo.write(objects.transformation.matrix.reshape(-1).detach().cpu().numpy())
            self.color_vbo.write(objects.appearance.color.reshape(-1).detach().cpu().numpy())
            self.confidence_vbo.write(objects.appearance.confidence.reshape(-1).detach().cpu().numpy())

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
        n_verts = min(self.n_verts - 1, self.ctx.info["GL_MAX_GEOMETRY_OUTPUT_COMPONENTS"] // 2 + 1)
        self.geometry_shader = f"""
            #version 330
            
            const int n_verts = {n_verts};
            const float pi = {pi};
            const float radius = 1;
            const float angle = 2 * pi / n_verts;
            
            layout (points) in;
            in vec4 v_color[];
            in mat3 v_mat[];
            layout ({"line_strip" if self.contours_only else "triangle_strip"}, max_vertices = {n_verts * 2 + 1}) out;
            out vec4 g_color;
            
            void main() {{
                for (int i = 0; i <= n_verts; i++)
                {{
                    float currentAngle = angle * i;
                    float x = radius * cos(currentAngle);
                    float y = radius * sin(currentAngle);
                    
                    vec2 position = (vec3(x, y, 1) * v_mat[0]).xy;
                    gl_Position = vec4(position * 2 - 1, 0, 1);
                    g_color = v_color[0];
                    EmitVertex();
                    
                    {"" if self.contours_only else
                    "position = (vec3(0, 0, 1) * v_mat[0]).xy;"
                    "gl_Position = vec4(position * 2 - 1, 0, 1);"
                    "g_color = v_color[0];"
                    "EmitVertex();"}
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
