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
#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>

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
};

struct SceeneData {
    glm::vec3 Center;
};

void outputMatrix(glm::mat4 matrix) {
    for (int i = 0; i < 4; i++) {
        std::cout << matrix[i][0] << ", " << matrix[i][1] << ", " << matrix[i][2] << ", " << matrix[i][3] << std::endl;
    }
}

glm::mat4 mixMat4(const glm::mat4& mat1, const glm::mat4& mat2, float t) {
    glm::mat4 result;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            result[i][j] = glm::mix(mat1[i][j], mat2[i][j], t);
        }
    }
    return result;
}

glm::mat4 matMixer(glm::mat4 mat1, glm::mat4 mat2, uint32_t startFrame, uint32_t endFrame, uint32_t currentFrame, std::string easingType) {
    glm::mat4 result;
    if (easingType == "linear") {
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

struct Transform {
    glm::mat4 matrix;

    //コンストラクタ
    Transform(glm::vec3 pos , glm::vec3 rot , glm::vec3 scale ) {
        glm::mat4 translate = glm::translate(pos);
        glm::mat4 rotate = glm::rotate(glm::mat4(1.0f), rot.x, glm::vec3(1, 0, 0)) * glm::rotate(glm::mat4(1.0f), rot.y, glm::vec3(0, 1, 0)) * glm::rotate(glm::mat4(1.0f), rot.z, glm::vec3(0, 0, 1));
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
    uint32_t lowerBoundFrameIndex = 0;

    glm::mat4 getMatrix(uint32_t currentFrame) {
       if(currentFrame == keyframes.at(lowerBoundFrameIndex).startFrame) {
           return modelMatrices.at(lowerBoundFrameIndex).matrix;
           lowerBoundFrameIndex++;
       }
       else{
            return matMixer(modelMatrices.at(lowerBoundFrameIndex).matrix, modelMatrices.at(lowerBoundFrameIndex+1).matrix  //mat1, mat2
            , keyframes.at(lowerBoundFrameIndex).startFrame, keyframes.at(lowerBoundFrameIndex+1).startFrame, currentFrame  //startFrame, endFrame, currentFrame
            , keyframes.at(lowerBoundFrameIndex).easingtype);   //easingType
       } 
    } 
};

struct Camera {
    std::vector<glm::mat4> viewMatrices;
    glm::mat4 projectionMatrices = glm::perspective(glm::radians(45.0f), (float)screenWidth / (float)screenHeight, 0.1f, 100.0f);
    std::vector<KeyFrame> keyframes;
    uint32_t lowerBoundFrameIndex = 0;

    glm::mat4 getMatrix(uint32_t currentFrame) {
       if(currentFrame == keyframes.at(lowerBoundFrameIndex).startFrame) {
           return viewMatrices.at(lowerBoundFrameIndex);
           lowerBoundFrameIndex++;
       }
       else{
            return matMixer(viewMatrices.at(lowerBoundFrameIndex), viewMatrices.at(lowerBoundFrameIndex+1)  //mat1, mat2
            , keyframes.at(lowerBoundFrameIndex).startFrame, keyframes.at(lowerBoundFrameIndex+1).startFrame, currentFrame  //startFrame, endFrame, currentFrame
            , keyframes.at(lowerBoundFrameIndex).easingtype);   //easingType
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
        for (int i = 0; i < numVertices; ++i) {
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
            obj.vertices.push_back(vertex);
        }
        // インデックスデータの読み込み
        for (int i = 0; i < numIndices; i++) {
            std::getline(file, line);
            std::stringstream indexStream(line);
            std::string s;
            std::vector<std::string> indexData;
            while (std::getline(indexStream, s, ',')) {
                indexData.push_back(s);
            }
            for(int j = 1;j<indexData.size();j++){
                obj.indices.push_back(std::stoi(indexData.at(j)));
                //std::cout << "Index: " << obj.indices.at(i*3+j-1) << std::endl;
            }
        }

        // キーフレームデータの読み込み
        for (int i = 0; i < numKeyFrames; ++i) {
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
            glm::vec3 pos, rot, scale = glm::vec3(1, 1, 1);
            glm::mat4 viewMatrix;

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

            Transform transform = Transform(pos, rot, scale);
            viewMatrix = glm::inverse(transform.matrix);

            camera.keyframes.push_back(keyframe);
            camera.viewMatrices.push_back(viewMatrix);
            
            std::cout << "KeyFrame: " << keyframe.startFrame << ", " << pos.x << ", " << pos.y << ", " << pos.z << ", " << rot.x << ", " << rot.y << ", " << rot.z << ", " << scale.x << ", " << scale.y << ", " << scale.z << ", " << keyframe.easingtype << std::endl;
        }
    return camera;
};

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

int main() {
    auto requiredLayers = { "VK_LAYER_KHRONOS_validation" };

    //GLFWの初期化
    if (!glfwInit())
        return -1;

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

    vk::UniqueDevice device = physicalDevice.createDeviceUnique(devCreateInfo);

    //キューの取得
    vk::Queue graphicsQueue = device->getQueue(graphicsQueueFamilyIndex, 0);

    //オブジェクトをCSVファイルから読み込む

    std::vector<Object> objects = loadObjectsFromCSV("C:/Users/enjoy/Documents/VisualStudioCode/vulkan/sessions2024/output.csv");

    Camera camera = loadCameraFromCSV("C:/Users/enjoy/Documents/VisualStudioCode/vulkan/sessions2024/camera.csv");

    // オブジェクトの確認
    
    for (const auto& obj : objects) {
        std::cout << "Object with " << obj.vertices.size() << " vertices and " << obj.indices.size() << " indices." << std::endl;
        for (int i = 0; i < obj.modelMatrices.size(); i++) {
            std::cout << "Keyframe at " << obj.keyframes.at(i).startFrame << " with easing type " << obj.keyframes.at(i).easingtype << std::endl;
            outputMatrix(obj.modelMatrices.at(i).matrix);
        }
    }
    
    // カメラの確認
    std::cout << camera.viewMatrices.size() << " view matrices." << std::endl;
    for(int i = 0; i < camera.viewMatrices.size(); i++) {
        std::cout << "Keyframe at " << camera.keyframes.at(i).startFrame << " with easing type " << camera.keyframes.at(i).easingtype << std::endl;
        outputMatrix(camera.viewMatrices.at(i));
    }

    //頂点バッファの作成
    
    vk::BufferCreateInfo vertBufferCreateInfo;
    vertBufferCreateInfo.size = sizeof(Vertex) * vertices.size();
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
        stagingBufferCreateInfo.size = sizeof(Vertex) * vertices.size();
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

        std::memcpy(stagingBufMem, vertices.data(), stagingBufMemReq.size);

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
    indexBufferCreateInfo.size = sizeof(uint16_t) * indices.size();
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
        stagingBufferCreateInfo.size = sizeof(uint16_t) * indices.size();
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

        std::memcpy(pStagingBufMem, indices.data(), stagingBufMemReq.size);

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
        bufCopy.size = sizeof(uint16_t) * indices.size();

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
    
    vk::BufferCreateInfo uniformBufferCreateInfo;
    uniformBufferCreateInfo.size = sizeof(sceneData);
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

    std::memcpy(pUniformBufMem, &sceneData, uniformBufMemReq.size);

    vk::MappedMemoryRange uniformBufMemoryRange;
    uniformBufMemoryRange.memory = uniformBufMemory.get();
    uniformBufMemoryRange.offset = 0;
    uniformBufMemoryRange.size = uniformBufMemReq.size;

    device->flushMappedMemoryRanges({ uniformBufMemoryRange });

    device->unmapMemory(uniformBufMemory.get());
    
    //デスクリプタセットの作成
    
    vk::DescriptorSetLayoutBinding descSetLayoutBinding[1];
    descSetLayoutBinding[0].binding = 0;
    descSetLayoutBinding[0].descriptorType = vk::DescriptorType::eUniformBuffer;
    descSetLayoutBinding[0].descriptorCount = 1;
    descSetLayoutBinding[0].stageFlags = vk::ShaderStageFlagBits::eVertex;

    vk::DescriptorSetLayoutCreateInfo descSetLayoutCreateInfo{};
    descSetLayoutCreateInfo.bindingCount = 1;
    descSetLayoutCreateInfo.pBindings = descSetLayoutBinding;

    vk::UniqueDescriptorSetLayout descSetLayout = device->createDescriptorSetLayoutUnique(descSetLayoutCreateInfo);

    //デスクリプタプールの作成

    vk::DescriptorPoolSize descPoolSize[1];
    descPoolSize[0].type = vk::DescriptorType::eUniformBuffer;
    descPoolSize[0].descriptorCount = 1;

    vk::DescriptorPoolCreateInfo descPoolCreateInfo;
    descPoolCreateInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
    descPoolCreateInfo.poolSizeCount = 1;
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

    vk::WriteDescriptorSet writeDescSet;
    writeDescSet.dstSet = descSets[0].get();
    writeDescSet.dstBinding = 0;
    writeDescSet.dstArrayElement = 0;
    writeDescSet.descriptorType = vk::DescriptorType::eUniformBuffer;

    vk::DescriptorBufferInfo descBufInfo[1];
    descBufInfo[0].buffer = uniformBuf.get();
    descBufInfo[0].offset = 0;
    descBufInfo[0].range = sizeof(sceneData);

    writeDescSet.descriptorCount = 1;
    writeDescSet.pBufferInfo = descBufInfo;

    device->updateDescriptorSets({ writeDescSet }, {});
    
    //頂点入力バインディングデスクリプション

    vk::VertexInputBindingDescription vertexBindingDescription[1];
    vertexBindingDescription[0].binding = 0;
    vertexBindingDescription[0].stride = sizeof(Vertex);
    vertexBindingDescription[0].inputRate = vk::VertexInputRate::eVertex;

    //頂点入力アトリビュートデスクリプション

    vk::VertexInputAttributeDescription vertexInputDescription[3];
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
   
    //レンダーパスの作成

    vk::AttachmentDescription attachments[1];
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

    vk::SubpassDescription subpasses[1];
    subpasses[0].pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
    subpasses[0].colorAttachmentCount = 1;
    subpasses[0].pColorAttachments = subpass0_attachmentRefs;

    vk::RenderPassCreateInfo renderpassCreateInfo;
    renderpassCreateInfo.attachmentCount = 1;
    renderpassCreateInfo.pAttachments = attachments;
    renderpassCreateInfo.subpassCount = 1;
    renderpassCreateInfo.pSubpasses = subpasses;
    renderpassCreateInfo.dependencyCount = 0;
    renderpassCreateInfo.pDependencies = nullptr;

    vk::UniqueRenderPass renderpass = device->createRenderPassUnique(renderpassCreateInfo);

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

    //パイプラインの作成

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
    vertexInputInfo.vertexAttributeDescriptionCount = 3;
    vertexInputInfo.pVertexAttributeDescriptions = vertexInputDescription;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = vertexBindingDescription;

    vk::PipelineInputAssemblyStateCreateInfo inputAssembly;
    inputAssembly.topology = vk::PrimitiveTopology::eTriangleList;
    inputAssembly.primitiveRestartEnable = false;

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

    vk::PipelineColorBlendAttachmentState blendattachment[1];
    blendattachment[0].colorWriteMask =
        vk::ColorComponentFlagBits::eA |
        vk::ColorComponentFlagBits::eR |
        vk::ColorComponentFlagBits::eG |
        vk::ColorComponentFlagBits::eB;
    blendattachment[0].blendEnable = false;

    vk::PipelineColorBlendStateCreateInfo blend;
    blend.logicOpEnable = false;
    blend.attachmentCount = 1;
    blend.pAttachments = blendattachment;

    //デスクリプタセットレイアウトをパイプラインに設定
    auto pipelineDescSetLayouts = { descSetLayout.get() };

    vk::PipelineLayoutCreateInfo layoutCreateInfo;
    layoutCreateInfo.setLayoutCount = pipelineDescSetLayouts.size();
    layoutCreateInfo.pSetLayouts = pipelineDescSetLayouts.begin();

    vk::UniquePipelineLayout pipelineLayout = device->createPipelineLayoutUnique(layoutCreateInfo);

    //シェーダーステージの設定
    vk::PipelineShaderStageCreateInfo shaderStage[2];
    shaderStage[0].stage = vk::ShaderStageFlagBits::eVertex;
    shaderStage[0].module = vertShader.get();
    shaderStage[0].pName = "main";
    shaderStage[1].stage = vk::ShaderStageFlagBits::eFragment;
    shaderStage[1].module = fragShader.get();
    shaderStage[1].pName = "main";

    vk::GraphicsPipelineCreateInfo pipelineCreateInfo;
    pipelineCreateInfo.pViewportState = &viewportState;
    pipelineCreateInfo.pVertexInputState = &vertexInputInfo;
    pipelineCreateInfo.pInputAssemblyState = &inputAssembly;
    pipelineCreateInfo.pRasterizationState = &rasterizer;
    pipelineCreateInfo.pMultisampleState = &multisample;
    pipelineCreateInfo.pColorBlendState = &blend;
    pipelineCreateInfo.layout = pipelineLayout.get();
    pipelineCreateInfo.renderPass = renderpass.get();
    pipelineCreateInfo.subpass = 0;
    pipelineCreateInfo.stageCount = 2;
    pipelineCreateInfo.pStages = shaderStage;

    vk::UniquePipeline pipeline = device->createGraphicsPipelineUnique(nullptr, pipelineCreateInfo).value;

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
        vk::ImageView frameBufAttachments[1];
        frameBufAttachments[0] = swapchainImageViews[i].get();

        vk::FramebufferCreateInfo frameBufCreateInfo;
        frameBufCreateInfo.width = surfaceCapabilities.currentExtent.width;
        frameBufCreateInfo.height = surfaceCapabilities.currentExtent.height;
        frameBufCreateInfo.layers = 1;
        frameBufCreateInfo.renderPass = renderpass.get();
        frameBufCreateInfo.attachmentCount = 1;
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

    //メインループ
    int64_t frameCount = 0;

    while (!glfwWindowShouldClose(window)) {
        auto frameStartTime = std::chrono::high_resolution_clock::now();//フレームの開始時間を記録
        glfwPollEvents();

        //aquireNextImageの前にレンダリングが終わるまで待機
        device->waitForFences({ imgRenderedFence.get() }, VK_TRUE, UINT64_MAX);
        device->resetFences({ imgRenderedFence.get() });

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

        vk::ClearValue clearVal[1];
        clearVal[0].color.float32[0] = 0.0f;
        clearVal[0].color.float32[1] = 0.0f;
        clearVal[0].color.float32[2] = 0.0f;
        clearVal[0].color.float32[3] = 1.0f;

        vk::RenderPassBeginInfo renderpassBeginInfo;
        renderpassBeginInfo.renderPass = renderpass.get();
        renderpassBeginInfo.framebuffer = swapchainFramebufs[imgIndex].get();
        renderpassBeginInfo.renderArea = vk::Rect2D({ 0,0 }, { screenWidth, screenHeight });
        renderpassBeginInfo.clearValueCount = 1;
        renderpassBeginInfo.pClearValues = clearVal;

        cmdBufs[0]->beginRenderPass(renderpassBeginInfo, vk::SubpassContents::eInline);

            // ここでサブパス0番の処理

        cmdBufs[0]->bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline.get());
        cmdBufs[0]->bindVertexBuffers(0, { vertBuf.get() }, { 0 });
        cmdBufs[0]->bindIndexBuffer(indexBuf.get(), 0, vk::IndexType::eUint16);
        cmdBufs[0]->bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout.get(), 0, { descSets[0].get() }, {});   //デスクリプタセットのバインド

        cmdBufs[0]->drawIndexed(indices.size(), 1, 0, 0, 0);

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
    glfwTerminate();
    return 0;
}