#version 450
#extension GL_ARB_separate_shader_objects : enable

/*layout(set = 0, binding = 0) uniform SceneData {
    vec3 center;
} sceneData;
*/

layout(set = 0, binding = 0) uniform MVPData {
    //mat4 model;
    mat4 view;
    mat4 projection;
} MVPMatrices;

layout(set = 0, binding = 1) uniform OBJData{
    mat4 model[3];
} objData;

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in uint inObjectID;

layout(location = 0) out vec3 geomColor;
layout(location = 1) out vec3 geomPos;

void main() {
    mat4 worldView = MVPMatrices.view * objData.model[inObjectID];
    gl_Position = MVPMatrices.projection * MVPMatrices.view * objData.model[inObjectID] * vec4(inPos, 1.0);    

    //debugPrintfEXT("inPos: %f %f %f\n", inPos.x, inPos.y, inPos.z);

    //gl_Position = vec4(inPos, 1.0);
    geomColor = (inColor) * inObjectID + vec3(0.349, 0.0, 1.0) * (1 - inObjectID);
    geomPos =  (objData.model[inObjectID] * vec4(inPos, 1.0)).xyz ;
}