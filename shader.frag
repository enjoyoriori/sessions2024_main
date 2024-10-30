#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform MVPData {
    //mat4 model;
    mat4 view;
    mat4 projection;
    vec3 cameraPos;
} MVPMatrices;

layout(location = 0) in vec3 fragmentNormal;
layout(location = 1) in vec3 fragPosition; 
layout(set = 0, binding = 2) uniform sampler2D depthTexture; // デプスバッファのサンプラー
layout(location = 0) out vec4 outColor;

// レイマーチングのパラメータ
const int MAX_STEPS = 50;
const float MAX_DISTANCE = 100.0;
const float STEP_SIZE = 0.1;

vec3 ScreenSpaceRayMarch(vec3 startPos, vec3 dir) {
    float distanceTraveled = 0.0;

    for (int i = 0; i < MAX_STEPS; i++) {
        vec3 currentPos = startPos + dir * distanceTraveled;

        // 画面座標に変換
        vec4 clipSpacePos = MVPMatrices.projection * MVPMatrices.view * vec4(currentPos, 1.0);
        vec2 ndc = clipSpacePos.xy / clipSpacePos.w;
        vec2 uv = ndc * 0.5 + 0.5;

        // スクリーン外に出たら終了
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
            break;
        }

        // 現在のスクリーン位置からワールド空間位置を取得
        vec3 scenePos = texture(depthTexture, uv).rgb;

        // ヒット判定：現在の位置とテクスチャの位置の距離を確認
        if (distance(currentPos, scenePos) < STEP_SIZE) {
            return scenePos;
        }

        // ステップを進める
        distanceTraveled += STEP_SIZE;

        // 最大距離を超えた場合終了
        if (distanceTraveled > MAX_DISTANCE) {
            break;
        }
    }

    // ヒットしなかった場合、背景色（黒）を返す
    return vec3(0.0);
}



void main() {
    // rayの方向を求める
    vec3 viewDir = normalize(fragPosition - MVPMatrices.cameraPos);

    // 法線を正規化
    vec3 normal = fragmentNormal;
    
    // 環境光成分
    vec3 ambient = vec3(0.0, 0.0, 0.0);

    // レイマーチング
    vec3 hitPos = ScreenSpaceRayMarch(MVPMatrices.cameraPos, viewDir);

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
    outColor = vec4(depthTexture , 1.0);
    //outColor = vec4(normal.xyz,1);
}