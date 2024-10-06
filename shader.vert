#version 450
#extension GL_ARB_separate_shader_objects : enable

/*layout(set = 0, binding = 0) uniform SceneData {
    vec3 center;
} sceneData;
*/

layout(set = 0, binding = 0) uniform MVPData {
    mat4 model;
    mat4 view;
    mat4 projection;
} MVPMatrices;



layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec3 inNormal;

layout(location = 0) out vec3 fragmentColor;





void main() {
    mat4 worldView = MVPMatrices.view * MVPMatrices.model;
    gl_Position = MVPMatrices.projection * worldView * vec4(inPos, 1.0);    
    fragmentColor = inColor;
}