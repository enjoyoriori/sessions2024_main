#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform SceneData {
    vec3 center;
} sceneData;

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec3 inNormal;

layout(location = 0) out vec3 fragmentColor;


void main() {
    gl_Position = vec4(sceneData.center + inPos, 1.0);
    fragmentColor = inColor;
}