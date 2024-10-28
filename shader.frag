#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 fragmentColor;
layout(location = 0) out vec4 outColor;

void main() {
    // 法線を正規化
    vec3 normal = fragmentColor;
    
    // 環境光成分
    vec3 ambient = vec3(0.3843, 0.749, 0.8235);

    // ライト方向とカメラ方向のベクトル
    vec3 lightPos = vec3(1.0, 1.0, 1.0);
    vec3 viewPos = vec3(0.0, 0.0, 0.0);
    vec3 fragPosition = vec3(0.0, 0.0, 0.0);


    vec3 lightDir = normalize(lightPos - fragPosition);
    vec3 viewDir = normalize(viewPos - fragPosition);

    // 拡散光成分
    vec3 diffuseColor = vec3(1.0, 1.0, 1.0);
    vec3 specularColor = vec3(1.0, 1.0, 1.0);
    float shininess = 32.0;



    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * diffuseColor;

    // 鏡面光成分
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
    vec3 specular = spec * specularColor;

    // 最終カラーの計算
    outColor = vec4(ambient + diffuse + specular, 1.0);
}