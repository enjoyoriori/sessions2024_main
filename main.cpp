﻿#include <iostream>
#include <vector>
#include <map>
#include <fstream>
#include <filesystem>
#include <string>
#include <sstream>
#include <cstring>
#include <chrono>
#include <thread>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include <glm/gtx/log_base.hpp>
#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>
extern "C" {
    #define MINIAUDIO_IMPLEMENTATION
    #include "miniaudio.h"
}

// cmake .. -DCMAKE_TOOLCHAIN_FILE=C:\vcpkg/scripts/buildsystems/vcpkg.cmake
// cmake --build .

const uint32_t screenWidth = 1920;
const uint32_t screenHeight = 1080;

uint32_t frameCount = 0;

bool memoryChecker(vk::PhysicalDeviceMemoryProperties memProps, vk::MemoryRequirements memReq, vk::MemoryAllocateInfo& allocInfo) {
    bool suitableMemoryTypeFound = false;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if (memReq.memoryTypeBits & (1 << i)) {
            allocInfo.memoryTypeIndex = i;
            suitableMemoryTypeFound = true;
            break;
        }
    }
    if (!suitableMemoryTypeFound) {
        std::cerr << "適切なメモリタイプが存在しません。" << std::endl;
    }
        return suitableMemoryTypeFound;
}

bool memoryChecker(vk::PhysicalDeviceMemoryProperties memProps, vk::MemoryRequirements memReq, vk::MemoryAllocateInfo& allocInfo, vk::MemoryPropertyFlagBits memPropsFlag) {
    bool suitableMemoryTypeFound = false;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if (memReq.memoryTypeBits & (1 << i) && (memProps.memoryTypes[i].propertyFlags & memPropsFlag)) {
            allocInfo.memoryTypeIndex = i;
            suitableMemoryTypeFound = true;
            break;
        }
    }
    if (!suitableMemoryTypeFound) {
        std::cerr << "適切なメモリタイプが存在しません。" << std::endl;
    }
        return suitableMemoryTypeFound;
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
}

struct Vertex {
    glm::vec3 pos;
    glm::vec3 color;
    glm::vec3 normal;
    uint32_t objectIndex;
};

struct SceeneData {
    //glm::mat4 modelMatrix; モデル行列は別で送る
    glm::mat4 viewMatrix;
    glm::mat4 projectionMatrix;
    glm::vec3 cameraPos;
    glm::vec3 lightPos[3];
};

void outputMatrix(glm::mat4 matrix) {
    for (int i = 0; i < 4; i++) {
        std::cout << matrix[i][0] << ", " << matrix[i][1] << ", " << matrix[i][2] << ", " << matrix[i][3] << std::endl;
    }
}

glm::mat4 mixMat4(const glm::mat4& mat1, const glm::mat4& mat2, float t) {
    // 1. 行列からスケール、回転、位置を抽出する
    glm::vec3 scale1, scale2;
    glm::quat rotation1, rotation2;
    glm::vec3 translation1, translation2;
    glm::vec3 skew;
    glm::vec4 perspective;

    // mat1から成分を抽出
    glm::decompose(mat1, scale1, rotation1, translation1, skew, perspective);

    // mat2から成分を抽出
    glm::decompose(mat2, scale2, rotation2, translation2, skew, perspective);

    // 2. スケールの対数補間
    glm::vec3 safeScale1 = scale1;
    glm::vec3 safeScale2 = scale2;

    glm::vec3 logScale1 = glm::log(safeScale1);
    glm::vec3 logScale2 = glm::log(safeScale2);

    glm::vec3 interpolatedLogScale = glm::mix(logScale1, logScale2, t);
    glm::vec3 interpolatedScale = glm::exp(interpolatedLogScale);

    // 3. 回転の球面線形補間 (Slerp)
    glm::quat interpolatedRotation = glm::slerp(rotation1, rotation2, t);

    // 4. 位置の線形補間
    glm::vec3 interpolatedTranslation = glm::mix(translation1, translation2, t);

    // 5. 補間したスケール、回転、位置を使って新しい変換行列を生成
    glm::mat4 scaleMatrix = glm::scale(interpolatedScale);
    glm::mat4 rotationMatrix = glm::mat4_cast(interpolatedRotation);  // クォータニオンから行列に変換
    glm::mat4 translationMatrix = glm::translate(interpolatedTranslation);

    // 6. 新しい行列を組み立てる
    glm::mat4 resultMatrix = translationMatrix * rotationMatrix * scaleMatrix;

    return resultMatrix;
}

glm::mat4 matMixer(glm::mat4 mat1, glm::mat4 mat2, uint32_t startFrame, uint32_t endFrame, uint32_t currentFrame, std::string easingType) {
    glm::mat4 result;
    if (easingType == "LINEAR") {
        float t = (float)(currentFrame - startFrame) / (endFrame - startFrame);
        result = mixMat4(mat1, mat2, t);
    }
    else if(easingType == "CONSTANT") {
        if (currentFrame >= startFrame && currentFrame < endFrame) {
            result = mat1;
        }
        else {
            result = mat2;
        }
    }
    else if(easingType == "BEZIER") {
        float t = (float)(currentFrame - startFrame) / (endFrame - startFrame);
        float t2 = 3 * t * t - 2 * t * t * t;
        result = mixMat4(mat1, mat2, t2);
    }
    return result;
}

void playSoundInThread(ma_engine* pEngine, const char* filePath) {
    ma_result result;
    ma_sound sound;

    // 音声オブジェクトの初期化
    result = ma_sound_init_from_file(pEngine, filePath, 0, NULL, NULL, &sound);
    if (result != MA_SUCCESS) {
        std::cerr << "Failed to initialize sound." << std::endl;
        return;
    }

    // 音声ファイルを再生
    result = ma_sound_start(&sound);
    if (result != MA_SUCCESS) {
        std::cerr << "Failed to play sound." << std::endl;
        ma_sound_uninit(&sound);
        return;
    } else {
        std::cout << "Playing sound: " << filePath << std::endl;
    }

    // 再生が終了するまで待機
    while (ma_sound_is_playing(&sound)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    // 音声オブジェクトの終了
    ma_sound_uninit(&sound);
}

struct Transform {
    glm::mat4 matrix;

    //コンストラクタ
    Transform(glm::vec3 pos , glm::vec3 rot , glm::vec3 scale ) {
        glm::mat4 translate = glm::translate(pos);

        glm::mat4 rotateX = glm::rotate(glm::mat4(1.0f), rot.x, glm::vec3(1, 0, 0));
        glm::mat4 rotateY = glm::rotate(glm::mat4(1.0f), rot.y, glm::vec3(0, 1, 0));
        glm::mat4 rotateZ = glm::rotate(glm::mat4(1.0f), rot.z, glm::vec3(0, 0, 1));
        glm::mat4 rotate = rotateX * rotateY * rotateZ; // ZYXの順で回転を適用

        glm::mat4 scaleMatrix = glm::scale(scale);
        
        matrix = translate * rotate * scaleMatrix;
        outputMatrix(translate);
        outputMatrix(rotate);
        outputMatrix(scaleMatrix);
        outputMatrix(matrix);
    }
};

struct KeyFrame {
    uint32_t startFrame;
    std::string easingtype;
};

struct Object {
    std::vector<Vertex> vertices;
    std::vector<uint16_t> indices;
    std::vector<Transform> modelMatrices;
    std::vector<KeyFrame> keyframes;
    uint32_t upperBoundFrameIndex = 0;

    glm::mat4 getMatrix(uint32_t currentFrame) {
       if(currentFrame == keyframes.at(upperBoundFrameIndex).startFrame) {
            int i = upperBoundFrameIndex;
            upperBoundFrameIndex++;
            return modelMatrices.at(i).matrix;
       }
       else{
            std::cout << keyframes.at(upperBoundFrameIndex).startFrame << currentFrame << upperBoundFrameIndex << std::endl;
            return matMixer(modelMatrices.at(upperBoundFrameIndex-1).matrix, modelMatrices.at(upperBoundFrameIndex).matrix  //mat1, mat2
            , keyframes.at(upperBoundFrameIndex-1).startFrame, keyframes.at(upperBoundFrameIndex).startFrame, currentFrame  //startFrame, endFrame, currentFrame
            , keyframes.at(upperBoundFrameIndex-1).easingtype);   //easingType
       } 
    } 
};

struct Camera {
    std::vector<glm::mat4> viewMatrices;
    glm::mat4 projectionMatrices = glm::perspective(glm::radians(39.6f), (float)screenWidth / (float)screenHeight, 0.1f, 100.0f);

    std::vector<KeyFrame> keyframes;
    uint32_t upperBoundFrameIndex = 0;

    glm::mat4 getMatrix(uint32_t currentFrame) {
       if(currentFrame == keyframes.at(upperBoundFrameIndex).startFrame) {
            int i = upperBoundFrameIndex;
            upperBoundFrameIndex++;
            return viewMatrices.at(i);
       }
       else{
            std::cout << keyframes.at(upperBoundFrameIndex).startFrame << currentFrame << std::endl;
            return glm::inverse(matMixer(glm::inverse(viewMatrices.at(upperBoundFrameIndex-1)), glm::inverse(viewMatrices.at(upperBoundFrameIndex)) //mat1, mat2
            , keyframes.at(upperBoundFrameIndex-1).startFrame, keyframes.at(upperBoundFrameIndex).startFrame, currentFrame  //startFrame, endFrame, currentFrame
            , keyframes.at(upperBoundFrameIndex-1).easingtype));   //easingType
       } 
    } 
};

struct light {
    std::vector<glm::vec3> lightPos;
    std::vector<KeyFrame> keyframes;
    uint32_t upperBoundFrameIndex = 0;

    glm::vec3 getLightPos(uint32_t currentFrame) {
       if(currentFrame == keyframes.at(upperBoundFrameIndex).startFrame) {
            int i = upperBoundFrameIndex;
            upperBoundFrameIndex++;
            return lightPos.at(i);
        }
       else{
            std::cout << keyframes.at(upperBoundFrameIndex).startFrame << currentFrame << std::endl;
            return lightPos.at(upperBoundFrameIndex-1);
        }
    }
};


std::vector<Object> loadObjectsFromCSV(const std::string& filename) {
    std::ifstream file(filename);
    std::string line;
    std::vector<Object> objects;

    if (!file.is_open()) {
        std::cerr << "ファイルを開くことができませんでした: " << filename << std::endl;
        return objects;
    }

    std::string token;

    // 数値データの読み込み
    int numObjects;
    file >> numObjects;
    
    std::cout << "Object: " << numObjects << std::endl;
    fflush(stdout);

    for(int i = 0;i < numObjects; i++){

        int numVertices, numIndices, numKeyFrames;
        file >> numVertices >> numIndices >> numKeyFrames;
        std::cout << "Vertices: " << numVertices << ", Indices: " << numIndices << ", KeyFrames: " << numKeyFrames << std::endl;
        std::getline(file, line);
        Object obj;
        // 頂点データの読み込み
        for (int j = 0; j < numVertices; ++j) {
            std::getline(file, line);
            std::stringstream vertexStream(line);
            std::string s;
            std::vector<std::string> vertexData;
            while (std::getline(vertexStream, s, ',')) {
                vertexData.push_back(s);
            }
            Vertex vertex;
            vertex.pos = glm::vec3(std::stof(vertexData.at(1)), std::stof(vertexData.at(2)), std::stof(vertexData.at(3)));
            //std::cout << "Vertex: " << vertex.pos.x << ", " << vertex.pos.y << ", " << vertex.pos.z << std::endl;
            vertex.color = glm::vec3(1.0,0.0,0.0);//色の設定
            vertex.objectIndex = i;
            obj.vertices.push_back(vertex);
        }
        // インデックスデータの読み込み
        for (int j = 0; j < numIndices; j++) {
            std::getline(file, line);
            std::stringstream indexStream(line);
            std::string s;
            std::vector<std::string> indexData;
            while (std::getline(indexStream, s, ',')) {
                indexData.push_back(s);
            }
            for(int k = 1;k<indexData.size();k++){
                obj.indices.push_back(std::stoi(indexData.at(k)));
                //std::cout << "Index: " << obj.indices.at(i*3+j-1) << std::endl;
            }
        }

        // キーフレームデータの読み込み
        for (int j = 0; j < numKeyFrames; ++j) {
            std::getline(file, line);
            std::stringstream keyframeStream(line);
            KeyFrame keyframe;
            glm::vec3 pos, rot, scale = glm::vec3(1, 1, 1);

            std::getline(keyframeStream, token, ',');keyframe.startFrame = std::stoi(token);

            std::getline(keyframeStream, token, ',');   pos.x = std::stof(token);
            std::getline(keyframeStream, token, ',');   pos.y = std::stof(token);
            std::getline(keyframeStream, token, ',');   pos.z = std::stof(token);

            std::getline(keyframeStream, token, ',');   rot.x = std::stof(token);
            std::getline(keyframeStream, token, ',');   rot.y = std::stof(token);
            std::getline(keyframeStream, token, ',');   rot.z = std::stof(token);

            std::getline(keyframeStream, token, ',');   scale.x = std::stof(token);
            std::getline(keyframeStream, token, ',');   scale.y = std::stof(token);
            std::getline(keyframeStream, token, ',');   scale.z = std::stof(token);

            std::getline(keyframeStream, token, ',');   keyframe.easingtype = token;

            obj.keyframes.push_back(keyframe);
            Transform transform(pos, rot, scale);
            obj.modelMatrices.push_back(transform);
            
            std::cout << "KeyFrame: " << keyframe.startFrame << ", " << pos.x << ", " << pos.y << ", " << pos.z << ", " << rot.x << ", " << rot.y << ", " << rot.z << ", " << scale.x << ", " << scale.y << ", " << scale.z << ", " << keyframe.easingtype << std::endl;
        }

        objects.push_back(obj);
    }
    


    file.close();
    return objects;
}

Camera loadCameraFromCSV(const std::string& filename) {//viewMatrixの読み込み
    std::ifstream file(filename);
    std::string line;
    std::string token;
    Camera camera;

    while(std::getline(file, line)) {
            std::stringstream keyframeStream(line);
            KeyFrame keyframe;
            glm::vec3 eye, center, up = glm::vec3(1, 1, 1);
            glm::mat4 viewMatrix;

            std::getline(keyframeStream, token, ',');keyframe.startFrame = std::stoi(token);

            std::getline(keyframeStream, token, ',');   eye.x = std::stof(token);
            std::getline(keyframeStream, token, ',');   eye.y = std::stof(token);
            std::getline(keyframeStream, token, ',');   eye.z = std::stof(token);

            std::getline(keyframeStream, token, ',');   center.x = std::stof(token);
            std::getline(keyframeStream, token, ',');   center.y = std::stof(token);
            std::getline(keyframeStream, token, ',');   center.z = std::stof(token);

            std::getline(keyframeStream, token, ',');   up.x = std::stof(token);
            std::getline(keyframeStream, token, ',');   up.y = std::stof(token);
            std::getline(keyframeStream, token, ',');   up.z = std::stof(token);

            std::getline(keyframeStream, token, ',');   keyframe.easingtype = token;

            viewMatrix = glm::lookAt(eye, center, up);

            camera.keyframes.push_back(keyframe);
            camera.viewMatrices.push_back(viewMatrix);
            
            std::cout << "KeyFrame: " << keyframe.startFrame << ", " << eye.x << ", " << eye.y << ", " << eye.z << ", " << center.x << ", " << center.y << ", " << center.z << ", " << up.x << ", " << up.y << ", " << up.z << ", " << keyframe.easingtype << std::endl;
        }
    return camera;
};

std::vector<light> loadLightsFromCSV(const std::string& filename) {
    std::ifstream file(filename);
    std::string line;
    std::vector<light> lights;

    if (!file.is_open()) {
        std::cerr << "ファイルを開くことができませんでした: " << filename << std::endl;
        return lights;
    }

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        light l;
        glm::vec3 pos;
        uint32_t frame;
        int objectIndex;

        std::getline(ss, token, ',');
        objectIndex = std::stoi(token);

        std::getline(ss, token, ',');
        frame = std::stoi(token);

        std::getline(ss, token, ',');
        pos.x = std::stof(token);

        std::getline(ss, token, ',');
        pos.y = std::stof(token);

        std::getline(ss, token, ',');
        pos.z = std::stof(token);

        if (lights.size() <= objectIndex) {
            lights.resize(objectIndex + 1);
        }

        lights[objectIndex].lightPos.push_back(pos);
        KeyFrame keyframe;
        keyframe.startFrame = frame;
        lights[objectIndex].keyframes.push_back(keyframe);
    }

    file.close();
    return lights;
}

/*
std::vector<Vertex> vertices = {
    Vertex{ glm::vec3(-0.5f, -0.5f, 0.0f ), glm::vec3( 0.0, 0.0, 1.0 ) },
    Vertex{ glm::vec3( 0.5f,  0.5f, 0.0f ), glm::vec3( 0.0, 1.0, 0.0 ) },
    Vertex{ glm::vec3(-0.5f,  0.5f, 0.0f ), glm::vec3( 1.0, 0.0, 0.0 ) },
    Vertex{ glm::vec3( 0.5f, -0.5f, 0.0f ), glm::vec3( 1.0, 1.0, 1.0 ) },
};

std::vector<uint16_t> indices = { 0, 1, 2, 1, 0, 3 };

SceeneData sceneData = {
    glm::vec3{0.3f, -0.2f, 0.0f}
};
*/


int main() {
    auto requiredLayers = { "VK_LAYER_KHRONOS_validation" };

    //GLFWの初期化
    if (!glfwInit())
        return -1;

    //エンジンの初期化
    ma_engine engine;
    if (ma_engine_init(NULL, &engine) != MA_SUCCESS) {
        return -1;
    }

    //インスタンスの作成

    uint32_t requiredExtensionsCount;
    const char** requiredExtensions = glfwGetRequiredInstanceExtensions(&requiredExtensionsCount);//GLFWが必要とする拡張を取得

    vk::InstanceCreateInfo instCreateInfo;

    instCreateInfo.enabledLayerCount = requiredLayers.size();
    instCreateInfo.ppEnabledLayerNames = requiredLayers.begin();
    instCreateInfo.enabledExtensionCount = requiredExtensionsCount;
    instCreateInfo.ppEnabledExtensionNames = requiredExtensions;

    vk::UniqueInstance instance = vk::createInstanceUnique(instCreateInfo);

    //ウィンドウの作成

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_DECORATED, GLFW_FALSE); // 縁のないウィンドウを作成

    GLFWwindow* window;
    window = glfwCreateWindow(screenWidth, screenHeight, "GLFW Test Window", NULL, NULL);
    if (!window) {
        const char* err;
        glfwGetError(&err);
        std::cout << err << std::endl;
        glfwTerminate();
        return -1;
    }

    VkSurfaceKHR c_surface;
    VkResult result = glfwCreateWindowSurface(instance.get(), window, nullptr, &c_surface);
    if (result != VK_SUCCESS) {
        const char* err;
        glfwGetError(&err);
        std::cout << err << std::endl;
        glfwTerminate();
        return -1;
    }

    //キーボード入力のコールバック関数の登録
    glfwSetKeyCallback(window, keyCallback);

    // コンストラクタでVulkan-Hpp(C++)形式に変換
    vk::UniqueSurfaceKHR surface{ c_surface, instance.get() };

    //物理デバイス

    std::vector<vk::PhysicalDevice> physicalDevices = instance->enumeratePhysicalDevices();
    vk::PhysicalDevice physicalDevice;

    //キューのあるデバイスを選択

    bool existsSuitablePhysicalDevice = false;
    uint32_t graphicsQueueFamilyIndex;

    for (size_t i = 0; i < physicalDevices.size(); i++) {
        std::vector<vk::QueueFamilyProperties> queueProps = physicalDevices[i].getQueueFamilyProperties();
        bool existsGraphicsQueue = false;

        for (size_t j = 0; j < queueProps.size(); j++) {
            if (queueProps[j].queueFlags & vk::QueueFlagBits::eGraphics && 
                physicalDevices[i].getSurfaceSupportKHR(j, surface.get())) {
                existsGraphicsQueue = true;
                graphicsQueueFamilyIndex = j;
                break;
            }
        }

        //スワップチェインの拡張がサポートされているか確認
        std::vector<vk::ExtensionProperties> extProps = physicalDevices[i].enumerateDeviceExtensionProperties();
        bool supportsSwapchainExtension = false;

        for (size_t j = 0; j < extProps.size(); j++) {
            if (std::string_view(extProps[j].extensionName.data()) == VK_KHR_SWAPCHAIN_EXTENSION_NAME) {
                supportsSwapchainExtension = true;
                break;
            }
        }

        if (existsGraphicsQueue && supportsSwapchainExtension) {
            physicalDevice = physicalDevices[i];
            existsSuitablePhysicalDevice = true;
            break;
        }
        // ジオメトリシェーダー機能がサポートされているか確認
        vk::PhysicalDeviceFeatures deviceFeatures = physicalDevices[i].getFeatures();
        bool supportsGeometryShader = deviceFeatures.geometryShader;

        if (existsGraphicsQueue && supportsSwapchainExtension && supportsGeometryShader) {
            physicalDevice = physicalDevices[i];
            existsSuitablePhysicalDevice = true;
            break;
        }

        if (!existsSuitablePhysicalDevice) {
            std::cerr << "使用可能な物理デバイスがありません。" << std::endl;
            return -1;
        }
    }

    if (!existsSuitablePhysicalDevice) {
        std::cerr << "使用可能な物理デバイスがありません。" << std::endl;
        return -1;
    }

    //論理デバイスの作成

    vk::DeviceCreateInfo devCreateInfo;

    auto devRequiredExtensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
    
    devCreateInfo.enabledExtensionCount = devRequiredExtensions.size();
    devCreateInfo.ppEnabledExtensionNames = devRequiredExtensions.begin();
    devCreateInfo.enabledLayerCount = requiredLayers.size();
    devCreateInfo.ppEnabledLayerNames = requiredLayers.begin();
    
    //キュー情報の作成
    vk::DeviceQueueCreateInfo queueCreateInfo[1];
    queueCreateInfo[0].queueFamilyIndex = graphicsQueueFamilyIndex;
    queueCreateInfo[0].queueCount = 1;
    float queuePriorities[1] = {1.0f};
    queueCreateInfo[0].pQueuePriorities = queuePriorities;

    devCreateInfo.pQueueCreateInfos = queueCreateInfo;
    devCreateInfo.queueCreateInfoCount = 1;

    // 物理デバイス機能の設定
    vk::PhysicalDeviceFeatures deviceFeatures;
    deviceFeatures.geometryShader = VK_TRUE; // ジオメトリシェーダー機能を有効にする

    devCreateInfo.pEnabledFeatures = &deviceFeatures; // 機能を論理デバイス作成情報に追加

    vk::UniqueDevice device = physicalDevice.createDeviceUnique(devCreateInfo);
    

    //キューの取得
    vk::Queue graphicsQueue = device->getQueue(graphicsQueueFamilyIndex, 0);

    //オブジェクトをCSVファイルから読み込む
    std::vector<Object> objects = loadObjectsFromCSV("../../output.csv");

    uint32_t vertexCount = 0,indexCount = 0;;

    for(const auto& obj : objects) {
        vertexCount += obj.vertices.size();
        indexCount += obj.indices.size();
    }

    //カメラをCSVファイルから読み込む
    Camera camera = loadCameraFromCSV("../../camera.csv");

    //ライトをCSVファイルから読み込む
    std::vector<light> lights = loadLightsFromCSV("../../light.csv");

    // オブジェクトの確認
    for (const auto& obj : objects) {
        std::cout << "Object with " << obj.vertices.size() << " vertices and " << obj.indices.size() << " indices." << std::endl;
        for (int i = 0; i < obj.modelMatrices.size(); i++) {
            std::cout << "Keyframe at " << obj.keyframes.at(i).startFrame << " with easing type " << obj.keyframes.at(i).easingtype << std::endl;
            outputMatrix(obj.modelMatrices.at(i).matrix);
        }
    }
    
    // カメラの確認
    for(int i = 0; i < camera.viewMatrices.size(); i++) {
        std::cout << "Keyframe at " << camera.keyframes.at(i).startFrame << " with easing type " << camera.keyframes.at(i).easingtype << std::endl;
        outputMatrix(camera.viewMatrices.at(i));
    }

    //頂点バッファの作成
    
    vk::BufferCreateInfo vertBufferCreateInfo;
    vertBufferCreateInfo.size = sizeof(Vertex) * vertexCount;
    vertBufferCreateInfo.usage = vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst;
    vertBufferCreateInfo.sharingMode = vk::SharingMode::eExclusive;

    vk::UniqueBuffer vertBuf = device->createBufferUnique(vertBufferCreateInfo);

    //頂点バッファのメモリ割り当て

    vk::MemoryRequirements vertBufMemReq = device->getBufferMemoryRequirements(vertBuf.get());

    vk::PhysicalDeviceMemoryProperties vertMemProps = physicalDevice.getMemoryProperties();

    vk::MemoryAllocateInfo vertBufMemAllocInfo;
    vertBufMemAllocInfo.allocationSize = vertBufMemReq.size;

    if(!memoryChecker(vertMemProps, vertBufMemReq, vertBufMemAllocInfo, vk::MemoryPropertyFlagBits::eDeviceLocal)) {
        return -1;
    }

    vk::UniqueDeviceMemory vertBufMemory = device->allocateMemoryUnique(vertBufMemAllocInfo);

    device->bindBufferMemory(vertBuf.get(), vertBufMemory.get(), 0);
    
    //頂点バッファのステージングバッファの作成
    {
        vk::BufferCreateInfo stagingBufferCreateInfo;
        stagingBufferCreateInfo.size = sizeof(Vertex) * vertexCount;
        stagingBufferCreateInfo.usage = vk::BufferUsageFlagBits::eTransferSrc;
        stagingBufferCreateInfo.sharingMode = vk::SharingMode::eExclusive;

        vk::UniqueBuffer stagingBuf = device->createBufferUnique(stagingBufferCreateInfo);

        //ステージングバッファのメモリ割り当て
        vk::PhysicalDeviceMemoryProperties memProps = physicalDevice.getMemoryProperties();

        vk::MemoryRequirements stagingBufMemReq = device->getBufferMemoryRequirements(stagingBuf.get());

        vk::MemoryAllocateInfo stagingBufMemAllocInfo;
        stagingBufMemAllocInfo.allocationSize = stagingBufMemReq.size;

        if (!memoryChecker(memProps, stagingBufMemReq, stagingBufMemAllocInfo, vk::MemoryPropertyFlagBits::eHostVisible)) {
            return -1;
        }

        vk::UniqueDeviceMemory stagingBufMemory = device->allocateMemoryUnique(stagingBufMemAllocInfo);

        device->bindBufferMemory(stagingBuf.get(), stagingBufMemory.get(), 0);

        //ステージングバッファのメモリマッピング

        void* stagingBufMem = device->mapMemory(stagingBufMemory.get(), 0, stagingBufMemReq.size);         //sizeof(Vertex) * vertices.size());でやったらエラーでた

        std::vector<Vertex> allVertices;
        for (const auto& object : objects) {
            allVertices.insert(allVertices.end(), object.vertices.begin(), object.vertices.end());
        }

        std::memcpy(stagingBufMem, allVertices.data(), stagingBufMemReq.size); //頂点データのコピー

        vk::MappedMemoryRange flushMemoryRange;
        flushMemoryRange.memory = stagingBufMemory.get();
        flushMemoryRange.offset = 0;
        flushMemoryRange.size = stagingBufMemReq.size;

        device->flushMappedMemoryRanges({ flushMemoryRange });

        device->unmapMemory(stagingBufMemory.get());

            //コマンドバッファの作成

        vk::CommandPoolCreateInfo tmpCmdPoolCreateInfo;
        tmpCmdPoolCreateInfo.queueFamilyIndex = graphicsQueueFamilyIndex;
        tmpCmdPoolCreateInfo.flags = vk::CommandPoolCreateFlagBits::eTransient;
        vk::UniqueCommandPool tmpCmdPool = device->createCommandPoolUnique(tmpCmdPoolCreateInfo);

        vk::CommandBufferAllocateInfo tmpCmdBufAllocInfo;
        tmpCmdBufAllocInfo.commandPool = tmpCmdPool.get();
        tmpCmdBufAllocInfo.commandBufferCount = 1;
        tmpCmdBufAllocInfo.level = vk::CommandBufferLevel::ePrimary;
        std::vector<vk::UniqueCommandBuffer> tmpCmdBufs = device->allocateCommandBuffersUnique(tmpCmdBufAllocInfo);

        //データ転送命令

        vk::BufferCopy bufCopy;
        bufCopy.srcOffset = 0;
        bufCopy.dstOffset = 0;
        bufCopy.size = stagingBufMemReq.size;

        vk::CommandBufferBeginInfo cmdBeginInfo;
        cmdBeginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;

        tmpCmdBufs[0]->begin(cmdBeginInfo);
        tmpCmdBufs[0]->copyBuffer(stagingBuf.get(), vertBuf.get(), {bufCopy});
        tmpCmdBufs[0]->end();

        //キューでの取得

        vk::CommandBuffer submitCmdBuf[1] = {tmpCmdBufs[0].get()};
        vk::SubmitInfo submitInfo;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = submitCmdBuf;

        graphicsQueue.submit({submitInfo});
        graphicsQueue.waitIdle();
    }
    //インデックスバッファの作成
    
    vk::BufferCreateInfo indexBufferCreateInfo;
    indexBufferCreateInfo.size = sizeof(uint16_t) * indexCount;
    indexBufferCreateInfo.usage = vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst;    // ここだけ注意
    indexBufferCreateInfo.sharingMode = vk::SharingMode::eExclusive;

    vk::UniqueBuffer indexBuf = device->createBufferUnique(indexBufferCreateInfo);

    vk::MemoryRequirements indexBufMemReq = device->getBufferMemoryRequirements(indexBuf.get());
        
    vk::MemoryAllocateInfo indexBufMemAllocInfo;
    indexBufMemAllocInfo.allocationSize = indexBufMemReq.size;

    if (!memoryChecker(physicalDevice.getMemoryProperties(), indexBufMemReq, indexBufMemAllocInfo, vk::MemoryPropertyFlagBits::eHostVisible)) {
        return -1;
    }

    vk::UniqueDeviceMemory indexBufMemory = device->allocateMemoryUnique(indexBufMemAllocInfo);

    device->bindBufferMemory(indexBuf.get(), indexBufMemory.get(), 0);
    
    //インデックスバッファのステージングバッファの作成
    {
        vk::BufferCreateInfo stagingBufferCreateInfo;
        stagingBufferCreateInfo.size = sizeof(uint16_t) * indexCount;
        stagingBufferCreateInfo.usage = vk::BufferUsageFlagBits::eTransferSrc;
        stagingBufferCreateInfo.sharingMode = vk::SharingMode::eExclusive;

        vk::UniqueBuffer stagingBuf = device->createBufferUnique(stagingBufferCreateInfo);

        vk::MemoryRequirements stagingBufMemReq = device->getBufferMemoryRequirements(stagingBuf.get());

        vk::MemoryAllocateInfo stagingBufMemAllocInfo;
        stagingBufMemAllocInfo.allocationSize = stagingBufMemReq.size;

        if (!memoryChecker(physicalDevice.getMemoryProperties(), stagingBufMemReq, stagingBufMemAllocInfo, vk::MemoryPropertyFlagBits::eHostVisible)) {
            return -1;
        }

        vk::UniqueDeviceMemory stagingBufMemory = device->allocateMemoryUnique(stagingBufMemAllocInfo);

        device->bindBufferMemory(stagingBuf.get(), stagingBufMemory.get(), 0);

        void *pStagingBufMem = device->mapMemory(stagingBufMemory.get(), 0, stagingBufMemReq.size);

        std::vector<uint16_t> allIndices;

        uint32_t vertexOffset = 0;

        for (const auto& object : objects) {

            // インデックスデータを追加
            for (const auto& index : object.indices) {
                allIndices.push_back(index + vertexOffset);
            }

            // オフセットを更新
            vertexOffset += static_cast<uint32_t>(object.vertices.size());
        }

        std::memcpy(pStagingBufMem, allIndices.data(), stagingBufMemReq.size);

        vk::MappedMemoryRange flushMemoryRange;
        flushMemoryRange.memory = stagingBufMemory.get();
        flushMemoryRange.offset = 0;
        flushMemoryRange.size = stagingBufMemReq.size;

        device->flushMappedMemoryRanges({flushMemoryRange});

        device->unmapMemory(stagingBufMemory.get());

        vk::CommandPoolCreateInfo tmpCmdPoolCreateInfo;
        tmpCmdPoolCreateInfo.queueFamilyIndex = graphicsQueueFamilyIndex;
        tmpCmdPoolCreateInfo.flags = vk::CommandPoolCreateFlagBits::eTransient;
        vk::UniqueCommandPool tmpCmdPool = device->createCommandPoolUnique(tmpCmdPoolCreateInfo);

        vk::CommandBufferAllocateInfo tmpCmdBufAllocInfo;
        tmpCmdBufAllocInfo.commandPool = tmpCmdPool.get();
        tmpCmdBufAllocInfo.commandBufferCount = 1;
        tmpCmdBufAllocInfo.level = vk::CommandBufferLevel::ePrimary;
        std::vector<vk::UniqueCommandBuffer> tmpCmdBufs = device->allocateCommandBuffersUnique(tmpCmdBufAllocInfo);

        vk::BufferCopy bufCopy;
        bufCopy.srcOffset = 0;
        bufCopy.dstOffset = 0;
        bufCopy.size = sizeof(uint16_t) * indexCount;

        vk::CommandBufferBeginInfo cmdBeginInfo;
        cmdBeginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;

        tmpCmdBufs[0]->begin(cmdBeginInfo);
        tmpCmdBufs[0]->copyBuffer(stagingBuf.get(), indexBuf.get(), {bufCopy});
        tmpCmdBufs[0]->end();

        vk::CommandBuffer submitCmdBuf[1] = {tmpCmdBufs[0].get()};
        vk::SubmitInfo submitInfo;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = submitCmdBuf;

        graphicsQueue.submit({submitInfo});
        graphicsQueue.waitIdle();
    }

    //ユニフォームバッファの作成

    SceeneData sceneData = {
        //glm::mat4(1.0f),
        camera.viewMatrices.at(0),
        camera.projectionMatrices
    };

    // デバイスのプロパティを取得
    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);
    VkDeviceSize minUniformBufferOffsetAlignment = deviceProperties.limits.minUniformBufferOffsetAlignment;
        
    // シーンデータのサイズ
    VkDeviceSize sceneDataSize = sizeof(SceeneData);

    // モデル行列のサイズ
    VkDeviceSize modelMatricesSize = sizeof(glm::mat4) * objects.size();

    //オフセットの調整

    VkDeviceSize alignedOffset = ((sceneDataSize + minUniformBufferOffsetAlignment -1) / minUniformBufferOffsetAlignment) * minUniformBufferOffsetAlignment;

    // 必要なメモリサイズを計算し、minUniformBufferOffsetAlignmentの倍数に調整
    VkDeviceSize totalSize = sceneDataSize + modelMatricesSize;
    VkDeviceSize alignedTotalSize = ((alignedOffset + modelMatricesSize + minUniformBufferOffsetAlignment - 1) / minUniformBufferOffsetAlignment) * minUniformBufferOffsetAlignment;

    vk::BufferCreateInfo uniformBufferCreateInfo;
    uniformBufferCreateInfo.size = alignedTotalSize;
    uniformBufferCreateInfo.usage = vk::BufferUsageFlagBits::eUniformBuffer;
    uniformBufferCreateInfo.sharingMode = vk::SharingMode::eExclusive;

    vk::UniqueBuffer uniformBuf = device->createBufferUnique(uniformBufferCreateInfo);

    vk::MemoryRequirements uniformBufMemReq = device->getBufferMemoryRequirements(uniformBuf.get());

    vk::MemoryAllocateInfo uniformBufMemAllocInfo;
    vk::PhysicalDeviceMemoryProperties uniformMemProps = physicalDevice.getMemoryProperties();
    uniformBufMemAllocInfo.allocationSize = uniformBufMemReq.size;

    if (!memoryChecker(uniformMemProps, uniformBufMemReq, uniformBufMemAllocInfo, vk::MemoryPropertyFlagBits::eHostVisible)) {
        return -1;
    }

    vk::UniqueDeviceMemory uniformBufMemory = device->allocateMemoryUnique(uniformBufMemAllocInfo);

    device->bindBufferMemory(uniformBuf.get(), uniformBufMemory.get(), 0);

    //メモリマッピング

    void* pUniformBufMem = device->mapMemory(uniformBufMemory.get(), 0, uniformBufMemReq.size);

    //頂点入力バインディングデスクリプション

    vk::VertexInputBindingDescription vertexBindingDescription[1];
    vertexBindingDescription[0].binding = 0;
    vertexBindingDescription[0].stride = sizeof(Vertex);
    vertexBindingDescription[0].inputRate = vk::VertexInputRate::eVertex;

    //頂点入力アトリビュートデスクリプション

    vk::VertexInputAttributeDescription vertexInputDescription[4];
    vertexInputDescription[0].binding = 0;
    vertexInputDescription[0].location = 0;
    vertexInputDescription[0].format = vk::Format::eR32G32B32Sfloat;
    vertexInputDescription[0].offset = offsetof(Vertex, pos);

    vertexInputDescription[1].binding = 0;
    vertexInputDescription[1].location = 1;
    vertexInputDescription[1].format = vk::Format::eR32G32B32Sfloat;
    vertexInputDescription[1].offset = offsetof(Vertex, color);

    vertexInputDescription[2].binding = 0;
    vertexInputDescription[2].location = 2;
    vertexInputDescription[2].format = vk::Format::eR32G32B32Sfloat;
    vertexInputDescription[2].offset = offsetof(Vertex, normal);

    vertexInputDescription[3].binding = 0;
    vertexInputDescription[3].location = 3;
    vertexInputDescription[3].format = vk::Format::eR32Uint;
    vertexInputDescription[3].offset = offsetof(Vertex, objectIndex);
    
    
    //スワップチェインの作成
    

    vk::SurfaceCapabilitiesKHR surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR(surface.get());
    std::vector<vk::SurfaceFormatKHR> surfaceFormats = physicalDevice.getSurfaceFormatsKHR(surface.get());
    std::vector<vk::PresentModeKHR> surfacePresentModes = physicalDevice.getSurfacePresentModesKHR(surface.get());

    vk::SurfaceFormatKHR swapchainFormat = surfaceFormats[0];
    vk::PresentModeKHR swapchainPresentMode = surfacePresentModes[0];

    vk::SwapchainCreateInfoKHR swapchainCreateInfo;
    swapchainCreateInfo.surface = surface.get();
    swapchainCreateInfo.minImageCount = surfaceCapabilities.minImageCount + 1;
    swapchainCreateInfo.imageFormat = swapchainFormat.format;
    swapchainCreateInfo.imageColorSpace = swapchainFormat.colorSpace;
    swapchainCreateInfo.imageExtent = surfaceCapabilities.currentExtent;
    swapchainCreateInfo.imageArrayLayers = 1;
    swapchainCreateInfo.imageUsage = vk::ImageUsageFlagBits::eColorAttachment;
    swapchainCreateInfo.imageSharingMode = vk::SharingMode::eExclusive;
    swapchainCreateInfo.preTransform = surfaceCapabilities.currentTransform;
    swapchainCreateInfo.presentMode = swapchainPresentMode;
    swapchainCreateInfo.clipped = VK_TRUE;

    vk::UniqueSwapchainKHR swapchain = device->createSwapchainKHRUnique(swapchainCreateInfo);

    //スワップチェインのイメージの取得
    std::vector<vk::Image> swapchainImages = device->getSwapchainImagesKHR(swapchain.get());

    // 深度バッファのフォーマットを選択
    vk::Format depthFormat = vk::Format::eD32Sfloat;

    // 深度バッファ用のイメージを作成
    vk::ImageCreateInfo depthImageCreateInfo;
    depthImageCreateInfo.imageType = vk::ImageType::e2D;
    depthImageCreateInfo.extent = vk::Extent3D(screenWidth, screenHeight, 1);
    depthImageCreateInfo.mipLevels = 1;
    depthImageCreateInfo.arrayLayers = 1;
    depthImageCreateInfo.format = depthFormat;
    depthImageCreateInfo.tiling = vk::ImageTiling::eOptimal;
    depthImageCreateInfo.initialLayout = vk::ImageLayout::eUndefined;
    depthImageCreateInfo.usage =  vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled; // 修正: VK_IMAGE_USAGE_SAMPLED_BITを追加
    depthImageCreateInfo.sharingMode = vk::SharingMode::eExclusive;
    depthImageCreateInfo.samples = vk::SampleCountFlagBits::e1;

    vk::UniqueImage depthImage = device->createImageUnique(depthImageCreateInfo);

    // 深度バッファ用のメモリを割り当て
    vk::MemoryRequirements depthMemReq = device->getImageMemoryRequirements(depthImage.get());
    vk::MemoryAllocateInfo depthMemAllocInfo;
    depthMemAllocInfo.allocationSize = depthMemReq.size;

    if(!memoryChecker(physicalDevice.getMemoryProperties(), depthMemReq, depthMemAllocInfo, vk::MemoryPropertyFlagBits::eDeviceLocal)) {
        return -1;
    }

    vk::UniqueDeviceMemory depthImageMemory = device->allocateMemoryUnique(depthMemAllocInfo);
    device->bindImageMemory(depthImage.get(), depthImageMemory.get(), 0);

    // 深度バッファ用のイメージビューを作成
    vk::ImageViewCreateInfo depthImageViewCreateInfo;
    depthImageViewCreateInfo.image = depthImage.get();
    depthImageViewCreateInfo.viewType = vk::ImageViewType::e2D;
    depthImageViewCreateInfo.format = depthFormat;
    depthImageViewCreateInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eDepth;
    depthImageViewCreateInfo.subresourceRange.baseMipLevel = 0;
    depthImageViewCreateInfo.subresourceRange.levelCount = 1;
    depthImageViewCreateInfo.subresourceRange.baseArrayLayer = 0;
    depthImageViewCreateInfo.subresourceRange.layerCount = 1;

    vk::UniqueImageView depthImageView = device->createImageViewUnique(depthImageViewCreateInfo);

    //Gバッファの作成
    std::vector<vk::Format> gBufferFormats = {
        vk::Format::eR32G32B32A32Sfloat, // Position
        vk::Format::eR16G16B16A16Sfloat, // Normal
        vk::Format::eR8G8B8A8Unorm       // Albedo
    };

    std::vector<vk::UniqueImage> gBufferImages(gBufferFormats.size());
    std::vector<vk::UniqueDeviceMemory> gBufferMemories(gBufferFormats.size());
    std::vector<vk::UniqueImageView> gBufferImageViews(gBufferFormats.size());

    for (size_t i = 0; i < gBufferFormats.size(); ++i) {
        vk::ImageCreateInfo imageCreateInfo;
        imageCreateInfo.imageType = vk::ImageType::e2D;
        imageCreateInfo.format = gBufferFormats[i];
        imageCreateInfo.extent.width = screenWidth;
        imageCreateInfo.extent.height = screenHeight;
        imageCreateInfo.extent.depth = 1;
        imageCreateInfo.mipLevels = 1;
        imageCreateInfo.arrayLayers = 1;
        imageCreateInfo.samples = vk::SampleCountFlagBits::e1;
        imageCreateInfo.tiling = vk::ImageTiling::eOptimal;
        imageCreateInfo.usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled;
        imageCreateInfo.sharingMode = vk::SharingMode::eExclusive;
        imageCreateInfo.initialLayout = vk::ImageLayout::eUndefined;

        gBufferImages[i] = device->createImageUnique(imageCreateInfo);

        vk::MemoryRequirements memReq = device->getImageMemoryRequirements(gBufferImages[i].get());
        vk::MemoryAllocateInfo memAllocInfo;
        memAllocInfo.allocationSize = memReq.size;
        if(!memoryChecker(physicalDevice.getMemoryProperties(), memReq, memAllocInfo, vk::MemoryPropertyFlagBits::eDeviceLocal)) {
            return -1;
        }

        gBufferMemories[i] = device->allocateMemoryUnique(memAllocInfo);
        device->bindImageMemory(gBufferImages[i].get(), gBufferMemories[i].get(), 0);

        vk::ImageViewCreateInfo imageViewCreateInfo;
        imageViewCreateInfo.image = gBufferImages[i].get();
        imageViewCreateInfo.viewType = vk::ImageViewType::e2D;
        imageViewCreateInfo.format = gBufferFormats[i];
        imageViewCreateInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        imageViewCreateInfo.subresourceRange.baseMipLevel = 0;
        imageViewCreateInfo.subresourceRange.levelCount = 1;
        imageViewCreateInfo.subresourceRange.baseArrayLayer = 0;
        imageViewCreateInfo.subresourceRange.layerCount = 1;

        gBufferImageViews[i] = device->createImageViewUnique(imageViewCreateInfo);
    }
    // イメージの作成
/*
    vk::ImageCreateInfo imgCreateInfo;
    imgCreateInfo.imageType = vk::ImageType::e2D;
    imgCreateInfo.extent = vk::Extent3D(screenWidth, screenHeight, 1);
    imgCreateInfo.mipLevels = 1;
    imgCreateInfo.arrayLayers = 1;
    imgCreateInfo.format = vk::Format::eR8G8B8A8Unorm;
    imgCreateInfo.tiling = vk::ImageTiling::eOptimal;
    imgCreateInfo.initialLayout = vk::ImageLayout::eUndefined;
    imgCreateInfo.usage = vk::ImageUsageFlagBits::eColorAttachment;
    imgCreateInfo.sharingMode = vk::SharingMode::eExclusive;
    imgCreateInfo.samples = vk::SampleCountFlagBits::e1;

    vk::UniqueImage image = device->createImageUnique(imgCreateInfo);

    //イメージのメモリ割り当て

    vk::PhysicalDeviceMemoryProperties memProps = physicalDevice.getMemoryProperties();

    vk::MemoryRequirements imgMemReq = device->getImageMemoryRequirements(image.get());

    vk::MemoryAllocateInfo imgMemAllocInfo;
    imgMemAllocInfo.allocationSize = imgMemReq.size;

    bool suitableMemoryTypeFound = false;
    for (size_t i = 0; i < memProps.memoryTypeCount; i++) {
        if (imgMemReq.memoryTypeBits & (1 << i)) {
            imgMemAllocInfo.memoryTypeIndex = i;
            suitableMemoryTypeFound = true;
            break;
        }
    }

    if (!suitableMemoryTypeFound) {
        std::cerr << "使用可能なメモリタイプがありません。" << std::endl;
        return -1;
    }

    vk::UniqueDeviceMemory imgMem = device->allocateMemoryUnique(imgMemAllocInfo);

    device->bindImageMemory(image.get(), imgMem.get(), 0);

    //イメージビューの作成

    vk::ImageViewCreateInfo imgViewCreateInfo;
    imgViewCreateInfo.image = image.get();
    imgViewCreateInfo.viewType = vk::ImageViewType::e2D;
    imgViewCreateInfo.format = vk::Format::eR8G8B8A8Unorm;
    imgViewCreateInfo.components.r = vk::ComponentSwizzle::eIdentity;
    imgViewCreateInfo.components.g = vk::ComponentSwizzle::eIdentity;
    imgViewCreateInfo.components.b = vk::ComponentSwizzle::eIdentity;
    imgViewCreateInfo.components.a = vk::ComponentSwizzle::eIdentity;
    imgViewCreateInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
    imgViewCreateInfo.subresourceRange.baseMipLevel = 0;
    imgViewCreateInfo.subresourceRange.levelCount = 1;
    imgViewCreateInfo.subresourceRange.baseArrayLayer = 0;
    imgViewCreateInfo.subresourceRange.layerCount = 1;

    vk::UniqueImageView imgView = device->createImageViewUnique(imgViewCreateInfo);
*/
    
    //ジオメトリステージ用デスクリプタの作成
    //デスクリプタセットレイアウトの作成
    vk::DescriptorSetLayoutBinding descSetLayoutBinding[3];
    descSetLayoutBinding[0].binding = 0;
    descSetLayoutBinding[0].descriptorType = vk::DescriptorType::eUniformBuffer;
    descSetLayoutBinding[0].descriptorCount = 1;
    descSetLayoutBinding[0].stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eGeometry | vk::ShaderStageFlagBits::eFragment;

    descSetLayoutBinding[1].binding = 1;
    descSetLayoutBinding[1].descriptorType = vk::DescriptorType::eUniformBuffer;
    descSetLayoutBinding[1].descriptorCount = 1;
    descSetLayoutBinding[1].stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eGeometry | vk::ShaderStageFlagBits::eFragment;

    descSetLayoutBinding[2].binding = 2;
    descSetLayoutBinding[2].descriptorType = vk::DescriptorType::eCombinedImageSampler;
    descSetLayoutBinding[2].descriptorCount = 1;
    descSetLayoutBinding[2].stageFlags = vk::ShaderStageFlagBits::eFragment;
    descSetLayoutBinding[2].pImmutableSamplers = nullptr;

    vk::DescriptorSetLayoutCreateInfo descSetLayoutCreateInfo{};
    descSetLayoutCreateInfo.bindingCount = static_cast<uint32_t>(std::size(descSetLayoutBinding));
    descSetLayoutCreateInfo.pBindings = descSetLayoutBinding;

    vk::UniqueDescriptorSetLayout descSetLayout = device->createDescriptorSetLayoutUnique(descSetLayoutCreateInfo);

    //デスクリプタプールの作成
    vk::DescriptorPoolSize descPoolSize[2];
    descPoolSize[0].type = vk::DescriptorType::eUniformBuffer;
    descPoolSize[0].descriptorCount = 1;
    descPoolSize[1].type = vk::DescriptorType::eUniformBuffer;
    descPoolSize[1].descriptorCount = 1;

    vk::DescriptorPoolCreateInfo descPoolCreateInfo;
    descPoolCreateInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
    descPoolCreateInfo.poolSizeCount = 2;
    descPoolCreateInfo.pPoolSizes = descPoolSize;
    descPoolCreateInfo.maxSets = 1;

    vk::UniqueDescriptorPool descPool = device->createDescriptorPoolUnique(descPoolCreateInfo);

    //デスクリプタセットの作成
    vk::DescriptorSetAllocateInfo descSetAllocInfo;

    auto descSetLayouts = { descSetLayout.get() };

    descSetAllocInfo.descriptorPool = descPool.get();
    descSetAllocInfo.descriptorSetCount = descSetLayouts.size();
    descSetAllocInfo.pSetLayouts = descSetLayouts.begin();

    std::vector<vk::UniqueDescriptorSet> descSets = device->allocateDescriptorSetsUnique(descSetAllocInfo);

    //デスクリプタの更新
    vk::DescriptorBufferInfo descBufInfo[1];
    descBufInfo[0].buffer = uniformBuf.get();
    descBufInfo[0].offset = 0;
    descBufInfo[0].range = sceneDataSize;

    vk::DescriptorBufferInfo descBufInfoDynamic[1];
    descBufInfoDynamic[0].buffer = uniformBuf.get();
    descBufInfoDynamic[0].offset = alignedOffset;  // モデル行列はview/projection行列の後に配置
    descBufInfoDynamic[0].range = modelMatricesSize;    // 各モデル行列のサイズ

    vk::WriteDescriptorSet writeDescSet[2];
    writeDescSet[0].dstSet = descSets[0].get();
    writeDescSet[0].dstBinding = 0;
    writeDescSet[0].dstArrayElement = 0;
    writeDescSet[0].descriptorType = vk::DescriptorType::eUniformBuffer;
    writeDescSet[0].descriptorCount = 1;
    writeDescSet[0].pBufferInfo = descBufInfo;

    writeDescSet[1].dstSet = descSets[0].get();
    writeDescSet[1].dstBinding = 1;
    writeDescSet[1].dstArrayElement = 0;
    writeDescSet[1].descriptorType = vk::DescriptorType::eUniformBuffer;
    writeDescSet[1].descriptorCount = 1;
    writeDescSet[1].pBufferInfo = descBufInfoDynamic;

    device->updateDescriptorSets({ writeDescSet }, {});

    // サンプラーの作成
    vk::SamplerCreateInfo samplerCreateInfo{};
    samplerCreateInfo.magFilter = vk::Filter::eLinear; // マグニフィケーションフィルタリング
    samplerCreateInfo.minFilter = vk::Filter::eLinear; // ミニフィケーションフィルタリング
    samplerCreateInfo.addressModeU = vk::SamplerAddressMode::eRepeat; // U方向のアドレッシングモード
    samplerCreateInfo.addressModeV = vk::SamplerAddressMode::eRepeat; // V方向のアドレッシングモード
    samplerCreateInfo.addressModeW = vk::SamplerAddressMode::eRepeat; // W方向のアドレッシングモード
    samplerCreateInfo.anisotropyEnable = VK_FALSE; // アニソトロピックフィルタリングの無効化
    samplerCreateInfo.maxAnisotropy = 1.0f; // アニソトロピックフィルタリングの最大サンプル数
    samplerCreateInfo.borderColor = vk::BorderColor::eIntOpaqueBlack; // 境界色
    samplerCreateInfo.unnormalizedCoordinates = VK_FALSE; // 正規化された座標を使用
    samplerCreateInfo.compareEnable = VK_FALSE; // 比較モードの無効化
    samplerCreateInfo.compareOp = vk::CompareOp::eAlways; // 比較演算子
    samplerCreateInfo.mipmapMode = vk::SamplerMipmapMode::eLinear; // ミップマップモード
    samplerCreateInfo.mipLodBias = 0.0f; // ミップマップレベルオブディテールバイアス
    samplerCreateInfo.minLod = 0.0f; // 最小ミップマップレベル
    samplerCreateInfo.maxLod = 0.0f; // 最大ミップマップレベル

    vk::UniqueSampler sampler = device->createSamplerUnique(samplerCreateInfo);


    // シェーディングステージ用のデスクリプタセットレイアウトの作成
    std::vector<vk::DescriptorSetLayoutBinding> shadingLayoutBindings(gBufferFormats.size() + 1);
    for (size_t i = 0; i < gBufferFormats.size(); ++i) {
        shadingLayoutBindings[i].binding = static_cast<uint32_t>(i);
        shadingLayoutBindings[i].descriptorType = vk::DescriptorType::eCombinedImageSampler;
        shadingLayoutBindings[i].descriptorCount = 1;
        shadingLayoutBindings[i].stageFlags = vk::ShaderStageFlagBits::eFragment;
        shadingLayoutBindings[i].pImmutableSamplers = nullptr;
    }

    // ユニフォームバッファのバインディング
    shadingLayoutBindings[gBufferFormats.size()].binding = static_cast<uint32_t>(gBufferFormats.size());
    shadingLayoutBindings[gBufferFormats.size()].descriptorType = vk::DescriptorType::eUniformBuffer;
    shadingLayoutBindings[gBufferFormats.size()].descriptorCount = 1;
    shadingLayoutBindings[gBufferFormats.size()].stageFlags = vk::ShaderStageFlagBits::eFragment;
    shadingLayoutBindings[gBufferFormats.size()].pImmutableSamplers = nullptr;

    vk::DescriptorSetLayoutCreateInfo shadingLayoutCreateInfo{};
    shadingLayoutCreateInfo.bindingCount = static_cast<uint32_t>(shadingLayoutBindings.size());
    shadingLayoutCreateInfo.pBindings = shadingLayoutBindings.data();

    vk::UniqueDescriptorSetLayout shadingDescSetLayout = device->createDescriptorSetLayoutUnique(shadingLayoutCreateInfo);

    // シェーディングステージ用のデスクリプタプールの作成
    std::vector<vk::DescriptorPoolSize> shadingPoolSizes(gBufferFormats.size() + 1);
    for (size_t i = 0; i < gBufferFormats.size(); ++i) {
        shadingPoolSizes[i].type = vk::DescriptorType::eCombinedImageSampler;
        shadingPoolSizes[i].descriptorCount = 1;
    }

    // ユニフォームバッファのプールサイズ
    shadingPoolSizes[gBufferFormats.size()].type = vk::DescriptorType::eUniformBuffer;
    shadingPoolSizes[gBufferFormats.size()].descriptorCount = 1;

    vk::DescriptorPoolCreateInfo shadingPoolCreateInfo{};
    shadingPoolCreateInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
    shadingPoolCreateInfo.poolSizeCount = static_cast<uint32_t>(shadingPoolSizes.size());
    shadingPoolCreateInfo.pPoolSizes = shadingPoolSizes.data();
    shadingPoolCreateInfo.maxSets = 1;

    vk::UniqueDescriptorPool shadingDescPool = device->createDescriptorPoolUnique(shadingPoolCreateInfo);

    // シェーディングステージ用のデスクリプタセットの作成
    vk::DescriptorSetAllocateInfo shadingDescSetAllocInfo{};
    shadingDescSetAllocInfo.descriptorPool = shadingDescPool.get();
    shadingDescSetAllocInfo.descriptorSetCount = 1;
    shadingDescSetAllocInfo.pSetLayouts = &shadingDescSetLayout.get();

    vk::UniqueDescriptorSet shadingDescSet = std::move(device->allocateDescriptorSetsUnique(shadingDescSetAllocInfo).front());

    // シェーディングステージ用のデスクリプタセットの更新
    std::vector<vk::DescriptorImageInfo> gBufferImageInfos(gBufferFormats.size());
    for (size_t i = 0; i < gBufferFormats.size(); ++i) {
        gBufferImageInfos[i].imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        gBufferImageInfos[i].imageView = gBufferImageViews[i].get();
        gBufferImageInfos[i].sampler = sampler.get();
    }

    std::vector<vk::WriteDescriptorSet> shadingWriteDescSets(gBufferFormats.size() + 1);
    for (size_t i = 0; i < gBufferFormats.size(); ++i) {
        shadingWriteDescSets[i].dstSet = shadingDescSet.get();
        shadingWriteDescSets[i].dstBinding = static_cast<uint32_t>(i);
        shadingWriteDescSets[i].dstArrayElement = 0;
        shadingWriteDescSets[i].descriptorType = vk::DescriptorType::eCombinedImageSampler;
        shadingWriteDescSets[i].descriptorCount = 1;
        shadingWriteDescSets[i].pImageInfo = &gBufferImageInfos[i];
    }

    // ユニフォームバッファのデスクリプタセットの更新
    shadingWriteDescSets[gBufferFormats.size()].dstSet = shadingDescSet.get();
    shadingWriteDescSets[gBufferFormats.size()].dstBinding = static_cast<uint32_t>(gBufferFormats.size());
    shadingWriteDescSets[gBufferFormats.size()].dstArrayElement = 0;
    shadingWriteDescSets[gBufferFormats.size()].descriptorType = vk::DescriptorType::eUniformBuffer;
    shadingWriteDescSets[gBufferFormats.size()].descriptorCount = 1;
    shadingWriteDescSets[gBufferFormats.size()].pBufferInfo = descBufInfo;

    device->updateDescriptorSets(shadingWriteDescSets, nullptr);


    //ジオメトリステージ用レンダーパスの作成
/*
    vk::AttachmentDescription attachments[2];
    attachments[0].format = swapchainFormat.format; //謎
    attachments[0].samples = vk::SampleCountFlagBits::e1;
    attachments[0].loadOp = vk::AttachmentLoadOp::eClear;
    attachments[0].storeOp = vk::AttachmentStoreOp::eStore;
    attachments[0].stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
    attachments[0].stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
    attachments[0].initialLayout = vk::ImageLayout::eUndefined;
    attachments[0].finalLayout = vk::ImageLayout::ePresentSrcKHR;

    vk::AttachmentReference subpass0_attachmentRefs[1];
    subpass0_attachmentRefs[0].attachment = 0;
    subpass0_attachmentRefs[0].layout = vk::ImageLayout::eColorAttachmentOptimal;

    // 深度アタッチメントの設定
    attachments[1].format = depthFormat;
    attachments[1].samples = vk::SampleCountFlagBits::e1;
    attachments[1].loadOp = vk::AttachmentLoadOp::eClear;
    attachments[1].storeOp = vk::AttachmentStoreOp::eDontCare;
    attachments[1].stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
    attachments[1].stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
    attachments[1].initialLayout = vk::ImageLayout::eUndefined;
    attachments[1].finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

    vk::AttachmentReference depthAttachmentRef;
    depthAttachmentRef.attachment = 1;
    depthAttachmentRef.layout = vk::ImageLayout::eDepthStencilAttachmentOptimal;
   
    vk::SubpassDescription subpasses[1];
    subpasses[0].pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
    subpasses[0].colorAttachmentCount = 1;
    subpasses[0].pColorAttachments = subpass0_attachmentRefs;
    subpasses[0].pDepthStencilAttachment = &depthAttachmentRef;

    vk::RenderPassCreateInfo renderpassCreateInfo;
    renderpassCreateInfo.attachmentCount = static_cast<uint32_t>(std::size(attachments));
    renderpassCreateInfo.pAttachments = attachments;
    renderpassCreateInfo.subpassCount = 1;
    renderpassCreateInfo.pSubpasses = subpasses;
    renderpassCreateInfo.dependencyCount = 0;
    renderpassCreateInfo.pDependencies = nullptr;

    vk::UniqueRenderPass renderpass = device->createRenderPassUnique(renderpassCreateInfo);
*/
    
    // Gバッファ用のレンダーパスの作成
    std::vector<vk::AttachmentDescription> gBufferAttachments(gBufferFormats.size() + 1);   // +1は深度バッファ用
    for (size_t i = 0; i < gBufferFormats.size(); ++i) {
        gBufferAttachments[i].format = gBufferFormats[i];
        gBufferAttachments[i].samples = vk::SampleCountFlagBits::e1;
        gBufferAttachments[i].loadOp = vk::AttachmentLoadOp::eClear;
        gBufferAttachments[i].storeOp = vk::AttachmentStoreOp::eStore;
        gBufferAttachments[i].stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
        gBufferAttachments[i].stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
        gBufferAttachments[i].initialLayout = vk::ImageLayout::eUndefined;
        gBufferAttachments[i].finalLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    }

    gBufferAttachments[gBufferFormats.size()].format = depthFormat;
    gBufferAttachments[gBufferFormats.size()].samples = vk::SampleCountFlagBits::e1;
    gBufferAttachments[gBufferFormats.size()].loadOp = vk::AttachmentLoadOp::eClear;
    gBufferAttachments[gBufferFormats.size()].storeOp = vk::AttachmentStoreOp::eDontCare;
    gBufferAttachments[gBufferFormats.size()].stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
    gBufferAttachments[gBufferFormats.size()].stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
    gBufferAttachments[gBufferFormats.size()].initialLayout = vk::ImageLayout::eUndefined;
    gBufferAttachments[gBufferFormats.size()].finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

    std::vector<vk::AttachmentReference> gBufferColorReferences(gBufferFormats.size());
    for (size_t i = 0; i < gBufferFormats.size(); ++i) {
        gBufferColorReferences[i].attachment = i;
        gBufferColorReferences[i].layout = vk::ImageLayout::eColorAttachmentOptimal;
    }

    vk::AttachmentReference gBufferDepthReference;
    gBufferDepthReference.attachment = gBufferFormats.size();
    gBufferDepthReference.layout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

    vk::SubpassDescription gBufferSubpass;
    gBufferSubpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
    gBufferSubpass.colorAttachmentCount = static_cast<uint32_t>(gBufferColorReferences.size());
    gBufferSubpass.pColorAttachments = gBufferColorReferences.data();
    gBufferSubpass.pDepthStencilAttachment = &gBufferDepthReference;

    vk::RenderPassCreateInfo gBufferRenderPassInfo;
    gBufferRenderPassInfo.attachmentCount = static_cast<uint32_t>(gBufferAttachments.size());
    gBufferRenderPassInfo.pAttachments = gBufferAttachments.data();
    gBufferRenderPassInfo.subpassCount = 1;
    gBufferRenderPassInfo.pSubpasses = &gBufferSubpass;

    vk::UniqueRenderPass gBufferRenderPass = device->createRenderPassUnique(gBufferRenderPassInfo);

    // ライティングパス用のレンダーパスの作成
    vk::AttachmentDescription colorAttachment{};
    colorAttachment.format = swapchainFormat.format;
    colorAttachment.samples = vk::SampleCountFlagBits::e1;
    colorAttachment.loadOp = vk::AttachmentLoadOp::eClear;
    colorAttachment.storeOp = vk::AttachmentStoreOp::eStore;
    colorAttachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
    colorAttachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
    colorAttachment.initialLayout = vk::ImageLayout::eUndefined;
    colorAttachment.finalLayout = vk::ImageLayout::ePresentSrcKHR;

    vk::AttachmentReference colorAttachmentRef{};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = vk::ImageLayout::eColorAttachmentOptimal;

    vk::AttachmentDescription depthAttachment{};
    depthAttachment.format = depthFormat;
    depthAttachment.samples = vk::SampleCountFlagBits::e1;
    depthAttachment.loadOp = vk::AttachmentLoadOp::eClear;
    depthAttachment.storeOp = vk::AttachmentStoreOp::eDontCare;
    depthAttachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
    depthAttachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
    depthAttachment.initialLayout = vk::ImageLayout::eUndefined;
    depthAttachment.finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

    vk::AttachmentReference depthAttachmentRef{};
    depthAttachmentRef.attachment = 1;
    depthAttachmentRef.layout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

    vk::SubpassDescription subpass{};
    subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;
    subpass.pDepthStencilAttachment = &depthAttachmentRef;

    std::array<vk::AttachmentDescription, 2> attachments = { colorAttachment, depthAttachment };

    vk::RenderPassCreateInfo renderPassInfo{};
    renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
    renderPassInfo.pAttachments = attachments.data();
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;

    vk::UniqueRenderPass lightingRenderPass = device->createRenderPassUnique(renderPassInfo);

    // Gバッファ用フレームバッファの作成
    std::vector<vk::UniqueFramebuffer> gBufferFramebuffers(swapchainImages.size());

    for (size_t i = 0; i < swapchainImages.size(); ++i) {
        std::vector<vk::ImageView> attachments(gBufferFormats.size() + 1);
        for (size_t j = 0; j < gBufferFormats.size(); ++j) {
            attachments[j] = gBufferImageViews[j].get();
        }
        attachments[gBufferFormats.size()] = depthImageView.get();

        vk::FramebufferCreateInfo framebufferCreateInfo;
        framebufferCreateInfo.renderPass = gBufferRenderPass.get();
        framebufferCreateInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        framebufferCreateInfo.pAttachments = attachments.data();
        framebufferCreateInfo.width = screenWidth;
        framebufferCreateInfo.height = screenHeight;
        framebufferCreateInfo.layers = 1;

        gBufferFramebuffers[i] = device->createFramebufferUnique(framebufferCreateInfo);
    }

    //ジオメトリステージ用のシェーダーの準備
    //バーテックスシェーダーの読み込み
    std::string vertShaderPath = "../../shader.vert.spv";
    std::string fragShaderPath = "../../shader.frag.spv";

    size_t vertSpvFileSz = std::filesystem::file_size(vertShaderPath);

    std::ifstream vertSpvFile(vertShaderPath, std::ios_base::binary);

    std::vector<char> vertSpvFileData(vertSpvFileSz);
    vertSpvFile.read(vertSpvFileData.data(), vertSpvFileSz);

    //シェーダーモジュールの作成
    vk::ShaderModuleCreateInfo vertShaderCreateInfo;
    vertShaderCreateInfo.codeSize = vertSpvFileSz;
    vertShaderCreateInfo.pCode = reinterpret_cast<const uint32_t*>(vertSpvFileData.data());

    vk::UniqueShaderModule vertShader = device->createShaderModuleUnique(vertShaderCreateInfo);

    //ジオメトリシェーダーの読み込み
    std::string geomShaderPath = "../../shader.geom.spv";
    size_t geomSpvFileSz = std::filesystem::file_size(geomShaderPath);
    std::ifstream geomSpvFile(geomShaderPath, std::ios_base::binary);
    std::vector<char> geomSpvFileData(geomSpvFileSz);
    geomSpvFile.read(geomSpvFileData.data(), geomSpvFileSz);

    vk::ShaderModuleCreateInfo geomShaderCreateInfo;
    geomShaderCreateInfo.codeSize = geomSpvFileSz;
    geomShaderCreateInfo.pCode = reinterpret_cast<const uint32_t*>(geomSpvFileData.data());
    vk::UniqueShaderModule geomShader = device->createShaderModuleUnique(geomShaderCreateInfo);


    //フラグメントシェーダーの読み込み
    size_t fragSpvFileSz = std::filesystem::file_size(fragShaderPath);

    std::ifstream fragSpvFile(fragShaderPath, std::ios_base::binary);

    std::vector<char> fragSpvFileData(fragSpvFileSz);
    fragSpvFile.read(fragSpvFileData.data(), fragSpvFileSz);

    //シェーダーモジュールの作成

    vk::ShaderModuleCreateInfo fragShaderCreateInfo;
    fragShaderCreateInfo.codeSize = fragSpvFileSz;
    fragShaderCreateInfo.pCode = reinterpret_cast<const uint32_t*>(fragSpvFileData.data());

    vk::UniqueShaderModule fragShader = device->createShaderModuleUnique(fragShaderCreateInfo);

    //ジオメトリステージ用パイプラインの作成
    vk::Viewport viewports[1];
    viewports[0].x = 0.0;
    viewports[0].y = 0.0;
    viewports[0].minDepth = 0.0;
    viewports[0].maxDepth = 1.0;
    viewports[0].width = screenWidth;
    viewports[0].height = screenHeight;

    vk::Rect2D scissors[1];
    scissors[0].offset = vk::Offset2D{ 0, 0 };
    scissors[0].extent = vk::Extent2D{ screenWidth, screenHeight };

    vk::PipelineViewportStateCreateInfo viewportState;
    viewportState.viewportCount = 1;
    viewportState.pViewports = viewports;
    viewportState.scissorCount = 1;
    viewportState.pScissors = scissors;

    vk::PipelineVertexInputStateCreateInfo vertexInputInfo;
    vertexInputInfo.vertexAttributeDescriptionCount = 4;
    vertexInputInfo.pVertexAttributeDescriptions = vertexInputDescription;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = vertexBindingDescription;

    vk::PipelineInputAssemblyStateCreateInfo inputAssembly;
    inputAssembly.topology = vk::PrimitiveTopology::eTriangleList;
    inputAssembly.primitiveRestartEnable = false;

    // 深度ステンシルステートの設定
    vk::PipelineDepthStencilStateCreateInfo depthStencil;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = vk::CompareOp::eLess;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.stencilTestEnable = VK_FALSE;

    vk::PipelineRasterizationStateCreateInfo rasterizer;
    rasterizer.depthClampEnable = false;
    rasterizer.rasterizerDiscardEnable = false;
    rasterizer.polygonMode = vk::PolygonMode::eFill;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = vk::CullModeFlagBits::eBack;
    rasterizer.frontFace = vk::FrontFace::eClockwise;
    rasterizer.depthBiasEnable = false;

    vk::PipelineMultisampleStateCreateInfo multisample;
    multisample.sampleShadingEnable = false;
    multisample.rasterizationSamples = vk::SampleCountFlagBits::e1;

    // ジオメトリステージのカラーブレンドステートの設定
    vk::PipelineColorBlendAttachmentState geomColorBlendAttachment{};
    geomColorBlendAttachment.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
    geomColorBlendAttachment.blendEnable = VK_FALSE;

    std::array<vk::PipelineColorBlendAttachmentState, 3> geomColorBlendAttachments = { geomColorBlendAttachment, geomColorBlendAttachment, geomColorBlendAttachment };

    vk::PipelineColorBlendStateCreateInfo geomColorBlendState{};
    geomColorBlendState.attachmentCount = static_cast<uint32_t>(geomColorBlendAttachments.size());
    geomColorBlendState.pAttachments = geomColorBlendAttachments.data();

    //デスクリプタセットレイアウトをパイプラインに設定
    auto pipelineDescSetLayouts = { descSetLayout.get() };

    vk::PipelineLayoutCreateInfo layoutCreateInfo;
    layoutCreateInfo.setLayoutCount = pipelineDescSetLayouts.size();
    layoutCreateInfo.pSetLayouts = pipelineDescSetLayouts.begin();

    vk::UniquePipelineLayout pipelineLayout = device->createPipelineLayoutUnique(layoutCreateInfo);

    //シェーダーステージの設定
    vk::PipelineShaderStageCreateInfo shaderStage[3];
    shaderStage[0].stage = vk::ShaderStageFlagBits::eVertex;
    shaderStage[0].module = vertShader.get();
    shaderStage[0].pName = "main";

    shaderStage[1].stage = vk::ShaderStageFlagBits::eGeometry;
    shaderStage[1].module = geomShader.get();
    shaderStage[1].pName = "main";

    shaderStage[2].stage = vk::ShaderStageFlagBits::eFragment;
    shaderStage[2].module = fragShader.get();
    shaderStage[2].pName = "main";

    vk::GraphicsPipelineCreateInfo pipelineCreateInfo;
    pipelineCreateInfo.pViewportState = &viewportState;
    pipelineCreateInfo.pVertexInputState = &vertexInputInfo;
    pipelineCreateInfo.pInputAssemblyState = &inputAssembly;
    pipelineCreateInfo.pRasterizationState = &rasterizer;
    pipelineCreateInfo.pMultisampleState = &multisample;
    pipelineCreateInfo.pColorBlendState = &geomColorBlendState;
    pipelineCreateInfo.pDepthStencilState = &depthStencil;
    pipelineCreateInfo.layout = pipelineLayout.get();
    pipelineCreateInfo.renderPass = gBufferRenderPass.get();
    pipelineCreateInfo.subpass = 0;
    pipelineCreateInfo.stageCount = 3;
    pipelineCreateInfo.pStages = shaderStage;

    vk::UniquePipeline pipeline = device->createGraphicsPipelineUnique(nullptr, pipelineCreateInfo).value;

    //ライティングステージ用のシェーダーの準備
    // 頂点シェーダーの読み込み
    std::string lightVertShaderPath = "../../lighting_shader.vert.spv";
    size_t lightVertSpvFileSz = std::filesystem::file_size(lightVertShaderPath);

    std::ifstream lightVertSpvFile(lightVertShaderPath, std::ios_base::binary);

    std::vector<char> lightVertSpvFileData(lightVertSpvFileSz);
    lightVertSpvFile.read(lightVertSpvFileData.data(), lightVertSpvFileSz);

    // 頂点シェーダーモジュールの作成
    vk::ShaderModuleCreateInfo lightVertShaderCreateInfo;
    lightVertShaderCreateInfo.codeSize = lightVertSpvFileSz;
    lightVertShaderCreateInfo.pCode = reinterpret_cast<const uint32_t*>(lightVertSpvFileData.data());

    vk::UniqueShaderModule lightVertShader = device->createShaderModuleUnique(lightVertShaderCreateInfo);

    //フラグメントシェーダーの読み込み
    std::string lightFragShaderPath = "../../lighting_shader.frag.spv";
    size_t lightFragSpvFileSz = std::filesystem::file_size(lightFragShaderPath);

    std::ifstream lightFragSpvFile(lightFragShaderPath, std::ios_base::binary);

    std::vector<char> lightFragSpvFileData(lightFragSpvFileSz);
    lightFragSpvFile.read(lightFragSpvFileData.data(), lightFragSpvFileSz);

    //シェーダーモジュールの作成

    vk::ShaderModuleCreateInfo lightFragShaderCreateInfo;
    lightFragShaderCreateInfo.codeSize = lightFragSpvFileSz;
    lightFragShaderCreateInfo.pCode = reinterpret_cast<const uint32_t*>(lightFragSpvFileData.data());

    vk::UniqueShaderModule lightFragShader = device->createShaderModuleUnique(lightFragShaderCreateInfo);

    // シェーダーステージの設定
    vk::PipelineShaderStageCreateInfo lightingShaderStages[2];
    lightingShaderStages[0].stage = vk::ShaderStageFlagBits::eVertex;
    lightingShaderStages[0].module = lightVertShader.get(); // ライティングパス用の頂点シェーダー
    lightingShaderStages[0].pName = "main";

    lightingShaderStages[1].stage = vk::ShaderStageFlagBits::eFragment;
    lightingShaderStages[1].module = lightFragShader.get(); // ライティングパス用のフラグメントシェーダー
    lightingShaderStages[1].pName = "main";

    //ライティングステージ用のカラーブレンドステートの設定
    vk::PipelineColorBlendAttachmentState lightColorBlendAttachment{};
    lightColorBlendAttachment.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
    lightColorBlendAttachment.blendEnable = VK_FALSE;

    std::array<vk::PipelineColorBlendAttachmentState, 1> lightColorBlendAttachments = { lightColorBlendAttachment };

    vk::PipelineColorBlendStateCreateInfo lightColorBlendState{};
    lightColorBlendState.attachmentCount = static_cast<uint32_t>(lightColorBlendAttachments.size());
    lightColorBlendState.pAttachments = lightColorBlendAttachments.data();


    // パイプラインレイアウトの作成
    auto lightingPipelineDescSetLayouts = { shadingDescSetLayout.get() };

    vk::PipelineLayoutCreateInfo lightingLayoutCreateInfo;
    lightingLayoutCreateInfo.setLayoutCount = lightingPipelineDescSetLayouts.size();
    lightingLayoutCreateInfo.pSetLayouts = lightingPipelineDescSetLayouts.begin();

    vk::UniquePipelineLayout lightingPipelineLayout = device->createPipelineLayoutUnique(lightingLayoutCreateInfo);

    // パイプラインの作成
    vk::GraphicsPipelineCreateInfo lightingPipelineCreateInfo;
    lightingPipelineCreateInfo.pViewportState = &viewportState;
    lightingPipelineCreateInfo.pVertexInputState = &vertexInputInfo;
    lightingPipelineCreateInfo.pInputAssemblyState = &inputAssembly;
    lightingPipelineCreateInfo.pRasterizationState = &rasterizer;
    lightingPipelineCreateInfo.pMultisampleState = &multisample;
    lightingPipelineCreateInfo.pColorBlendState = &lightColorBlendState;
    lightingPipelineCreateInfo.pDepthStencilState = &depthStencil;
    lightingPipelineCreateInfo.layout = lightingPipelineLayout.get();
    lightingPipelineCreateInfo.renderPass = lightingRenderPass.get();
    lightingPipelineCreateInfo.subpass = 0;
    lightingPipelineCreateInfo.stageCount = 2; // シェーダーステージの数を2に設定
    lightingPipelineCreateInfo.pStages = lightingShaderStages;

    vk::UniquePipeline lightingPipeline = device->createGraphicsPipelineUnique(nullptr, lightingPipelineCreateInfo).value;

    //イメージビューの作成
    std::vector<vk::UniqueImageView> swapchainImageViews(swapchainImages.size());

    for (size_t i = 0; i < swapchainImages.size(); i++) {
        vk::ImageViewCreateInfo imgViewCreateInfo;
        imgViewCreateInfo.image = swapchainImages[i];
        imgViewCreateInfo.viewType = vk::ImageViewType::e2D;
        imgViewCreateInfo.format = swapchainFormat.format;
        imgViewCreateInfo.components.r = vk::ComponentSwizzle::eIdentity;
        imgViewCreateInfo.components.g = vk::ComponentSwizzle::eIdentity;
        imgViewCreateInfo.components.b = vk::ComponentSwizzle::eIdentity;
        imgViewCreateInfo.components.a = vk::ComponentSwizzle::eIdentity;
        imgViewCreateInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        imgViewCreateInfo.subresourceRange.baseMipLevel = 0;
        imgViewCreateInfo.subresourceRange.levelCount = 1;
        imgViewCreateInfo.subresourceRange.baseArrayLayer = 0;
        imgViewCreateInfo.subresourceRange.layerCount = 1;

        swapchainImageViews[i] = device->createImageViewUnique(imgViewCreateInfo);
    }

    //フレームバッファの作成
/*
    vk::ImageView frameBufAttachments[1];
    frameBufAttachments[0] = imgView.get();

    vk::FramebufferCreateInfo frameBufCreateInfo;
    frameBufCreateInfo.width = screenWidth;
    frameBufCreateInfo.height = screenHeight;
    frameBufCreateInfo.layers = 1;
    frameBufCreateInfo.renderPass = renderpass.get();
    frameBufCreateInfo.attachmentCount = 1;
    frameBufCreateInfo.pAttachments = frameBufAttachments;

    vk::UniqueFramebuffer frameBuf = device->createFramebufferUnique(frameBufCreateInfo);
*/

    std::vector<vk::UniqueFramebuffer> swapchainFramebufs(swapchainImages.size());

    for (size_t i = 0; i < swapchainImages.size(); i++) {
        vk::ImageView frameBufAttachments[2];
        frameBufAttachments[0] = swapchainImageViews[i].get();
        frameBufAttachments[1] = depthImageView.get();

        vk::FramebufferCreateInfo frameBufCreateInfo;
        frameBufCreateInfo.width = surfaceCapabilities.currentExtent.width;
        frameBufCreateInfo.height = surfaceCapabilities.currentExtent.height;
        frameBufCreateInfo.layers = 1;
        frameBufCreateInfo.renderPass = lightingRenderPass.get();
        frameBufCreateInfo.attachmentCount = static_cast<uint32_t>(std::size(attachments));
        frameBufCreateInfo.pAttachments = frameBufAttachments;

        swapchainFramebufs[i] = device->createFramebufferUnique(frameBufCreateInfo);
    }


    //コマンドプールの作成
    vk::CommandPoolCreateInfo cmdPoolCreateInfo;
    cmdPoolCreateInfo.queueFamilyIndex = graphicsQueueFamilyIndex;
    cmdPoolCreateInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;

    vk ::UniqueCommandPool cmdPool = device->createCommandPoolUnique(cmdPoolCreateInfo);

    //コマンドバッファの作成
    vk::CommandBufferAllocateInfo cmdBufAllocInfo;
    cmdBufAllocInfo.commandPool = cmdPool.get();
    cmdBufAllocInfo.commandBufferCount = 1;
    cmdBufAllocInfo.level = vk::CommandBufferLevel::ePrimary;

    std::vector<vk::UniqueCommandBuffer> cmdBufs = device->allocateCommandBuffersUnique(cmdBufAllocInfo);

    //フェンスの作成
    vk::FenceCreateInfo fenceCreateInfo;
    fenceCreateInfo.flags = vk::FenceCreateFlagBits::eSignaled;
    //vk::UniqueFence swapchainImgFence = device->createFenceUnique(fenceCreateInfo);
    vk::UniqueFence imgRenderedFence = device->createFenceUnique(fenceCreateInfo);

    //セマフォの作成

    vk::SemaphoreCreateInfo semaphoreCreateInfo;

    vk::UniqueSemaphore swapchainImgSemaphore, imgRenderedSemaphore;
    swapchainImgSemaphore = device->createSemaphoreUnique(semaphoreCreateInfo);
    imgRenderedSemaphore = device->createSemaphoreUnique(semaphoreCreateInfo);

    //サウンドファイルの再生
    ma_engine_set_volume(&engine, 1.0f);

    std::thread soundThread(playSoundInThread, &engine, "../../sound.mp3");

    //メインループ
    int64_t frameCount = 0;//start FrameCount

    while (!glfwWindowShouldClose(window) && frameCount < 4271) {
        auto frameStartTime = std::chrono::high_resolution_clock::now();//フレームの開始時間を記録
        glfwPollEvents();

        //aquireNextImageの前にレンダリングが終わるまで待機
        device->waitForFences({ imgRenderedFence.get() }, VK_TRUE, UINT64_MAX);
        device->resetFences({ imgRenderedFence.get() });

        //ユニフォームバッファの更新
        //sceneData.modelMatrix = glm::rotate(glm::mat4(1.0f), glm::radians(0.1f * frameCount), glm::vec3(0.0f, 1.0f, 0.0f));

        sceneData.viewMatrix = camera.getMatrix(frameCount);
        sceneData.projectionMatrix = camera.projectionMatrices;
        sceneData.projectionMatrix[1][1] *= -1; // Y軸反転
        sceneData.cameraPos = camera.getMatrix(frameCount)[3];
        
        std::vector<glm::vec3> lightPositions;
        for(int i = 0; i < lights.size(); i++){
            lightPositions.push_back(lights.at(i).getLightPos(frameCount));
            sceneData.lightPos[i] = lightPositions.at(i) ;
        }

        // シーンデータのコピー
        std::memcpy(pUniformBufMem, &sceneData, sceneDataSize);



        // モデル行列の更新
        std::vector<glm::mat4> modelMatrices(objects.size());
        for (int i = 0; i < objects.size(); i++) {
            modelMatrices.at(i) = objects.at(i).getMatrix(frameCount);
            std::cout << "modelMatrices.at(" << i << "): " << std::endl;
        }

        // モデル行列を調整したオフセットにコピー
        std::memcpy(static_cast<char*>(pUniformBufMem) + alignedOffset, modelMatrices.data(), modelMatricesSize);

        // メモリ範囲のフラッシュ
        vk::MappedMemoryRange uniformBufMemoryRange;
        uniformBufMemoryRange.memory = uniformBufMemory.get();
        uniformBufMemoryRange.offset = 0;
        uniformBufMemoryRange.size = uniformBufMemReq.size;

        device->flushMappedMemoryRanges({ uniformBufMemoryRange });


        vk::ResultValue acquireImgResult = device->acquireNextImageKHR(swapchain.get(), 1'000'000'000, swapchainImgSemaphore.get());
        if (acquireImgResult.result != vk::Result::eSuccess) {
            std::cerr << "次フレームの取得に失敗しました。" << std::endl;
            return -1;
        }
        uint32_t imgIndex = acquireImgResult.value;
        cmdBufs[0]->reset();

        //コマンドの作成    
        vk::CommandBufferBeginInfo cmdBeginInfo;
        cmdBufs[0]->begin(cmdBeginInfo);

        // Gバッファのクリア
        std::vector<vk::ClearValue> gBufferClearValues(gBufferFormats.size() + 1);
        for (size_t i = 0; i < gBufferFormats.size(); ++i) {
            gBufferClearValues[i].color = vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f});
        }
        gBufferClearValues[gBufferFormats.size()].depthStencil = vk::ClearDepthStencilValue(1.0f, 0);

        // Gバッファへのレンダリング
        vk::RenderPassBeginInfo gBufferRenderPassBeginInfo;
        gBufferRenderPassBeginInfo.renderPass = gBufferRenderPass.get();
        gBufferRenderPassBeginInfo.framebuffer = gBufferFramebuffers[imgIndex].get();
        gBufferRenderPassBeginInfo.renderArea.offset = vk::Offset2D{0, 0};
        gBufferRenderPassBeginInfo.renderArea.extent = vk::Extent2D{screenWidth, screenHeight};
        gBufferRenderPassBeginInfo.clearValueCount = static_cast<uint32_t>(gBufferClearValues.size());
        gBufferRenderPassBeginInfo.pClearValues = gBufferClearValues.data();

        cmdBufs[0]->beginRenderPass(gBufferRenderPassBeginInfo, vk::SubpassContents::eInline);

            // ここでサブパス0番の処理

        cmdBufs[0]->bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline.get());
        cmdBufs[0]->bindVertexBuffers(0, { vertBuf.get() }, { 0 });
        cmdBufs[0]->bindIndexBuffer(indexBuf.get(), 0, vk::IndexType::eUint16);
        cmdBufs[0]->bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout.get(), 0, { descSets[0].get() }, {});   //デスクリプタセットのバインド
        cmdBufs[0]->drawIndexed(indexCount, 1, 0, 0, 0);
        cmdBufs[0]->endRenderPass();

        // ライティングパスへのレンダリング
        std::array<vk::ClearValue, 2> lightingClearValues{};
        lightingClearValues[0].color = vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f});
        lightingClearValues[1].depthStencil = vk::ClearDepthStencilValue{1.0f, 0}; // 深度クリア値を修正

        vk::RenderPassBeginInfo lightingRenderPassBeginInfo;
        lightingRenderPassBeginInfo.renderPass = lightingRenderPass.get();
        lightingRenderPassBeginInfo.framebuffer = swapchainFramebufs[imgIndex].get();
        lightingRenderPassBeginInfo.renderArea.offset = vk::Offset2D{0, 0};
        lightingRenderPassBeginInfo.renderArea.extent = vk::Extent2D{screenWidth, screenHeight};
        lightingRenderPassBeginInfo.clearValueCount = 2;
        lightingRenderPassBeginInfo.pClearValues = lightingClearValues.data();

        cmdBufs[0]->beginRenderPass(lightingRenderPassBeginInfo, vk::SubpassContents::eInline);
        cmdBufs[0]->bindPipeline(vk::PipelineBindPoint::eGraphics, lightingPipeline.get());
        cmdBufs[0]->bindDescriptorSets(vk::PipelineBindPoint::eGraphics, lightingPipelineLayout.get(), 0, { shadingDescSet.get() }, {});
        cmdBufs[0]->draw(6, 1, 0, 0);
        cmdBufs[0]->endRenderPass();

        cmdBufs[0]->end();

        

        vk::CommandBuffer submitCmdBuf[1] = {cmdBufs[0].get()};
        vk::SubmitInfo submitInfo;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = submitCmdBuf;

        //待機するセマフォの指定
        vk::Semaphore renderwaitSemaphores[] = { swapchainImgSemaphore.get() };
        vk::PipelineStageFlags renderwaitStages[] = { vk::PipelineStageFlagBits::eColorAttachmentOutput };
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = renderwaitSemaphores;
        submitInfo.pWaitDstStageMask = renderwaitStages;

        //完了時にシグナルするセマフォの指定
        vk::Semaphore renderSignalSemaphores[] = { imgRenderedSemaphore.get() };
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = renderSignalSemaphores;


        graphicsQueue.submit({submitInfo},imgRenderedFence.get());

        vk::PresentInfoKHR presentInfo;

        auto presentSwapchains = { swapchain.get() };
        auto imgIndices = { imgIndex };

        presentInfo.swapchainCount = presentSwapchains.size();
        presentInfo.pSwapchains = presentSwapchains.begin();
        presentInfo.pImageIndices = imgIndices.begin();

        //待機するセマフォの指定
        vk::Semaphore presentWaitSemaphores[] = { imgRenderedSemaphore.get() };
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = presentWaitSemaphores;

        graphicsQueue.presentKHR(presentInfo);

        // フレームレートを30fpsに制御
        std::cout << "Frame: " << frameCount << std::endl;
        frameCount++;
        
        double targetFrameDuration = 1000.0 / 30.0; // 30fpsのためのフレーム時間（ミリ秒）

        while(std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - frameStartTime).count() < targetFrameDuration) {
            std::this_thread::sleep_for(std::chrono::milliseconds(0));
        }
        std::cout << "wateing time: " << std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - frameStartTime).count() << std::endl;
    }

    graphicsQueue.waitIdle();

    device->unmapMemory(uniformBufMemory.get());//メモリのアンマップ
    glfwTerminate();

    // サウンドスレッドの終了
    soundThread.join();
    ma_engine_uninit(&engine);
    return 0;
}