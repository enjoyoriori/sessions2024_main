#version 450
layout(location = 0) in vec3 vertColor[];  // 各頂点の入力色
layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

layout(location = 0) out vec3 geomColor;   // 各頂点の出力色

void main() {
    // 三角形の3頂点からベクトルを計算
    vec3 edge1 = gl_in[1].gl_Position.xyz - gl_in[0].gl_Position.xyz;
    vec3 edge2 = gl_in[2].gl_Position.xyz - gl_in[0].gl_Position.xyz;

    // 法線を計算し正規化
    vec3 normal = normalize(cross(edge1, edge2));

    for (int i = 0; i < 3; i++) {
        vec4 offsetPosition = gl_in[i].gl_Position;  // Y軸方向にオフセット
        gl_Position = gl_in[i].gl_Position;
        geomColor = normal;  // 入力色をそのまま出力
        EmitVertex();
    }
    EndPrimitive();
}
