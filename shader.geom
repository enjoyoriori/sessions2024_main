#version 450
layout(location = 0) in vec3 vertColor[];  // 各頂点の入力色
layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

layout(location = 0) out vec3 geomColor;   // 各頂点の出力色

void main() {
    // 各入力頂点を処理
    for (int i = 0; i < 3; i++) {
        vec4 offsetPosition = gl_in[i].gl_Position + vec4(0.0, 1, 10, 200);  // Y軸方向にオフセット
        gl_Position = offsetPosition;
        geomColor = vertColor[i];  // 入力色をそのまま出力
        EmitVertex();
    }
    EndPrimitive();
}