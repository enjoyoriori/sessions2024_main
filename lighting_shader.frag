#version 450

layout(set = 0, binding = 0) uniform sampler2D gPosition; // Gバッファの位置
layout(set = 0, binding = 1) uniform sampler2D gNormal;   // Gバッファの法線
layout(set = 0, binding = 2) uniform sampler2D gAlbedo;   // Gバッファのアルベド
layout(set = 0, binding = 3) uniform MVPData {
    //mat4 model;
    mat4 view;
    mat4 projection;
    vec3 cameraPos;
    vec3 lightPos[2];
} ubo;

layout(location = 0) in vec2 fragTexCoord; // スクリーン空間のテクスチャ座標
layout(location = 0) out vec4 outColor;    // 最終的な色


void main() {
    // Gバッファからデータを読み取る
    vec3 position = texture(gPosition, fragTexCoord).rgb;
    vec3 normal = normalize(texture(gNormal, fragTexCoord).rgb);
    vec3 albedo = texture(gAlbedo, fragTexCoord).rgb;

    // SSAO計算

    // 簡単なライティング計算（例：フォンシェーディング
    vec3 diffuse = vec3(0.0);
    vec3 specular = vec3(0.0);
    vec3 viewDir = normalize(ubo.cameraPos - position);

    for(int i = 0; i < 2; i++) {
        vec3 lightDir = normalize(ubo.lightPos[i] - position);
        float diff = max(dot(normal, lightDir), 0.0);
        diffuse += diff * albedo;

        vec3 reflectDir = reflect(-lightDir, normal);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
        specular += spec * vec3(1.0);
    }

    vec3 ambient = 0.1 * albedo;

    vec3 color = ambient + diffuse + specular;
    outColor = vec4(color, 1.0);
}