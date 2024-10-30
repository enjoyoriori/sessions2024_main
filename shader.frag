#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform MVPData {
    //mat4 model;
    mat4 view;
    mat4 projection;
    vec3 cameraPos;
} MVPMatrices;

layout(location = 0) in vec3 fragNormal;
layout(location = 1) in vec3 fragPosition; 
layout(set = 0, binding = 2) uniform sampler2D depthTexture; // デプスバッファのサンプラー

layout(location = 0) out vec4 outPosition; // Gバッファの位置
layout(location = 1) out vec4 outNormal;   // Gバッファの法線
layout(location = 2) out vec4 outAlbedo;   // Gバッファのアルベド

void main() {
    // rayの方向を求める
    vec3 viewDir = normalize(fragPosition - MVPMatrices.cameraPos);

    // 法線を求める
    vec3 normal = normalize(fragNormal);

    // ライト方向とカメラ方向のベクトル
    vec3 lightPos = vec3(1.0, 1.0, 2.0);
    vec3 viewPos = vec3(0.0, 0.0, 0.0);


    vec3 lightDir = normalize(lightPos - fragPosition);

    // 拡散光成分
    vec3 diffuseColor = vec3(1.0, 1.0, 1.0);
    float shininess = 32.0;

    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * diffuseColor;


    // 最終カラーの計算
    outNormal = vec4(normal, 1.0);
    outAlbedo = vec4(diffuse, 1.0);
    outPosition = vec4(fragPosition, 1.0);

}