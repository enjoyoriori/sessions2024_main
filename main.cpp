#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1

#include <iostream>
#include <vector>
#include <map>
#include <fstream>
#include <filesystem>
#include <string>
#include <sstream>
#include <cstring>
#include <chrono>
#include <thread>
#include <functional>
#include <limits>
#define NOMINMAX
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

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE


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

struct SceneData {
    //glm::mat4 modelMatrix; モデル行列は別で送る
    glm::mat4 viewMatrix;
    glm::mat4 projectionMatrix;
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
    /*
    while (ma_sound_is_playing(&sound)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    */
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
    std::vector<uint32_t> indices;
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

inline uint32_t getMemoryType(vk::PhysicalDevice physicalDevice,
                              vk::MemoryRequirements memoryRequirements,
                              vk::MemoryPropertyFlags memoryProperties) {
    auto physicalDeviceMemoryProperties = physicalDevice.getMemoryProperties();
    for (uint32_t i = 0; i < VK_MAX_MEMORY_TYPES; ++i) {
        if (memoryRequirements.memoryTypeBits & (1 << i)) {
            if ((physicalDeviceMemoryProperties.memoryTypes[i].propertyFlags &
                 memoryProperties) == memoryProperties) {
                return i;
            }
        }
    }

    std::cerr << "Failed to get memory type index.\n";
    std::abort();
}

inline void oneTimeSubmit(vk::Device device,
                          vk::CommandPool commandPool,
                          vk::Queue queue,
                          const std::function<void(vk::CommandBuffer)>& func) {
    // Allocate
    vk::CommandBufferAllocateInfo allocateInfo{};
    allocateInfo.setCommandPool(commandPool);
    allocateInfo.setLevel(vk::CommandBufferLevel::ePrimary);
    allocateInfo.setCommandBufferCount(1);
    auto commandBuffers = device.allocateCommandBuffersUnique(allocateInfo);

    // Record
    commandBuffers[0]->begin(vk::CommandBufferBeginInfo{});
    func(commandBuffers[0].get());
    commandBuffers[0]->end();

    // Submit
    vk::UniqueFence fence = device.createFenceUnique({});
    vk::SubmitInfo submitInfo{};
    submitInfo.setCommandBuffers(commandBuffers[0].get());
    queue.submit(submitInfo, fence.get());

    // Wait
    if (device.waitForFences(fence.get(), true,std::numeric_limits<uint64_t>::max()) != vk::Result::eSuccess) {
        std::cerr << "Failed to wait for fence.\n";
        std::abort();
    }
}

struct Buffer {
    vk::UniqueBuffer buffer;
    vk::UniqueDeviceMemory memory;
    vk::DeviceAddress address{};
    void init(vk::PhysicalDevice physicalDevice,
              vk::Device device,
              vk::DeviceSize size,
              vk::BufferUsageFlags usage,
              vk::MemoryPropertyFlags memoryProperty,
              const void* data = nullptr) {
        // Create buffer
        vk::BufferCreateInfo createInfo{};
        createInfo.setSize(size);
        createInfo.setUsage(usage);
        buffer = device.createBufferUnique(createInfo);

        // Allocate memory
        vk::MemoryAllocateFlagsInfo allocateFlags{};
        if (usage & vk::BufferUsageFlagBits::eShaderDeviceAddress) {
            allocateFlags.flags = vk::MemoryAllocateFlagBits::eDeviceAddress;
        }

        vk::MemoryRequirements memoryReq =
            device.getBufferMemoryRequirements(*buffer);
        uint32_t memoryType = getMemoryType(physicalDevice,  //
                                                    memoryReq, memoryProperty);
        vk::MemoryAllocateInfo allocateInfo{};
        allocateInfo.setAllocationSize(memoryReq.size);
        allocateInfo.setMemoryTypeIndex(memoryType);
        allocateInfo.setPNext(&allocateFlags);
        memory = device.allocateMemoryUnique(allocateInfo);
        // Bind buffer to memory
        device.bindBufferMemory(*buffer, *memory, 0);
        // Copy data
        if (data) {
            void* mappedPtr = device.mapMemory(*memory, 0, size);
            memcpy(mappedPtr, data, size);
            device.unmapMemory(*memory);
        }
        // Get address
        if (usage & vk::BufferUsageFlagBits::eShaderDeviceAddress) {
            vk::BufferDeviceAddressInfoKHR addressInfo{};
            addressInfo.setBuffer(*buffer);
            address = device.getBufferAddressKHR(&addressInfo);
        }
    }
};

struct AccelStruct {
    vk::UniqueAccelerationStructureKHR accel;
    Buffer buffer;
    void init(vk::PhysicalDevice physicalDevice,
          vk::Device device,
          vk::CommandPool commandPool,
          vk::Queue queue,
          vk::AccelerationStructureTypeKHR type,
          vk::AccelerationStructureGeometryKHR geometry,
          uint32_t primitiveCount) {
        // Get build info
        vk::AccelerationStructureBuildGeometryInfoKHR buildInfo{};
        buildInfo.setType(type);
        buildInfo.setMode(vk::BuildAccelerationStructureModeKHR::eBuild);
        buildInfo.setFlags(
            vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace);
        buildInfo.setGeometries(geometry);

        vk::AccelerationStructureBuildSizesInfoKHR buildSizes =
            device.getAccelerationStructureBuildSizesKHR(
                vk::AccelerationStructureBuildTypeKHR::eDevice, buildInfo,
                primitiveCount);
        // Create buffer for AS
        buffer.init(physicalDevice, device,
                    buildSizes.accelerationStructureSize,
                    vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR,
                    vk::MemoryPropertyFlagBits::eDeviceLocal);
        // Create AS
        vk::AccelerationStructureCreateInfoKHR createInfo{};
        createInfo.setBuffer(*buffer.buffer);
        createInfo.setSize(buildSizes.accelerationStructureSize);
        createInfo.setType(type);
        accel = device.createAccelerationStructureKHRUnique(createInfo);
        // Create scratch buffer
        Buffer scratchBuffer;
        scratchBuffer.init(physicalDevice, device, buildSizes.buildScratchSize,
                        vk::BufferUsageFlagBits::eStorageBuffer |
                        vk::BufferUsageFlagBits::eShaderDeviceAddress,
                        vk::MemoryPropertyFlagBits::eDeviceLocal);

        buildInfo.setDstAccelerationStructure(*accel);
        buildInfo.setScratchData(scratchBuffer.address);
        // Build
        vk::AccelerationStructureBuildRangeInfoKHR buildRangeInfo{};
        buildRangeInfo.setPrimitiveCount(primitiveCount);
        buildRangeInfo.setPrimitiveOffset(0);
        buildRangeInfo.setFirstVertex(0);
        buildRangeInfo.setTransformOffset(0);

        oneTimeSubmit(          //
            device, commandPool, queue,  //
            [&](vk::CommandBuffer commandBuffer) {
                commandBuffer.buildAccelerationStructuresKHR(buildInfo,
                                                            &buildRangeInfo);
            });
        // Get address
        vk::AccelerationStructureDeviceAddressInfoKHR addressInfo{};
        addressInfo.setAccelerationStructure(*accel);
        buffer.address = device.getAccelerationStructureAddressKHR(addressInfo);
    }
};

void createBLAS(Object& object, vk::PhysicalDevice physicalDevice,
                vk::Device device, vk::CommandPool commandPool,
                vk::Queue queue, AccelStruct& bottomAccel) {
    //頂点バッファの作成
    vk::BufferUsageFlags bufferUsage{
        vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR |
        vk::BufferUsageFlagBits::eShaderDeviceAddress};
    vk::MemoryPropertyFlags memoryProperty{
        vk::MemoryPropertyFlagBits::eHostVisible |
        vk::MemoryPropertyFlagBits::eHostCoherent};
    Buffer vertexBuffer;
    Buffer indexBuffer;
    vertexBuffer.init(physicalDevice, device,
                      object.vertices.size() * sizeof(Vertex),
                      bufferUsage, memoryProperty, object.vertices.data());
    indexBuffer.init(physicalDevice, device,
                      object.indices.size() * sizeof(uint32_t),
                      bufferUsage, memoryProperty, object.indices.data());
    // Create geometry
    vk::AccelerationStructureGeometryTrianglesDataKHR triangles{};
    triangles.setVertexFormat(vk::Format::eR32G32B32Sfloat);
    triangles.setVertexData(vertexBuffer.address);
    triangles.setVertexStride(sizeof(Vertex));
    triangles.setMaxVertex(static_cast<uint32_t>(object.vertices.size()));
    triangles.setIndexType(vk::IndexType::eUint32);
    triangles.setIndexData(indexBuffer.address);

    vk::AccelerationStructureGeometryKHR geometry{};
    geometry.setGeometryType(vk::GeometryTypeKHR::eTriangles);
    geometry.setGeometry({triangles});
    geometry.setFlags(vk::GeometryFlagBitsKHR::eOpaque);
    // Create and build BLAS
    uint32_t primitiveCount = static_cast<uint32_t>(object.indices.size() / 3);
    bottomAccel.init(physicalDevice, device, commandPool, queue,
                     vk::AccelerationStructureTypeKHR::eBottomLevel,
                     geometry, primitiveCount);
}

inline auto getRayTracingProps(vk::PhysicalDevice physicalDevice) {
    auto deviceProperties = physicalDevice.getProperties2<
        vk::PhysicalDeviceProperties2,
        vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();
    return deviceProperties
        .get<vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();
}

inline uint32_t alignUp(uint32_t size, uint32_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

void createShaderBindingTable(vk::PhysicalDevice physicalDevice, 
                              vk::RayGenRegion raygenRegion, 
                              vk::MissRegion missRegion,
                              vk::HitRegion hitRegion,) {
    // Get RT props
    vk::PhysicalDeviceRayTracingPipelinePropertiesKHR rtProperties =
        getRayTracingProps(physicalDevice);
    uint32_t handleSize = rtProperties.shaderGroupHandleSize;
    uint32_t handleAlignment = rtProperties.shaderGroupHandleAlignment;
    uint32_t baseAlignment = rtProperties.shaderGroupBaseAlignment;
    uint32_t handleSizeAligned =
        alignUp(handleSize, handleAlignment);

    // Set strides and sizes
    uint32_t raygenShaderCount = 1;  // raygen count must be 1
    uint32_t missShaderCount = 1;
    uint32_t hitShaderCount = 1;

    raygenRegion.setStride(alignUp(handleSizeAligned, baseAlignment));
    raygenRegion.setSize(raygenRegion.stride);

    missRegion.setStride(handleSizeAligned);
    missRegion.setSize(alignUp(missShaderCount * handleSizeAligned,
                                        baseAlignment));

    hitRegion.setStride(handleSizeAligned);
    hitRegion.setSize(alignUp(hitShaderCount * handleSizeAligned,
                                        baseAlignment));
    // Create SBT
    vk::DeviceSize sbtSize =
        raygenRegion.size + missRegion.size + hitRegion.size;
    sbt.init(physicalDevice, device, sbtSize,
             vk::BufferUsageFlagBits::eShaderBindingTableKHR |
             vk::BufferUsageFlagBits::eTransferSrc |
             vk::BufferUsageFlagBits::eShaderDeviceAddress,
             vk::MemoryPropertyFlagBits::eHostVisible |
             vk::MemoryPropertyFlagBits::eHostCoherent);
    // Get shader group handles
    uint32_t handleCount = raygenShaderCount + missShaderCount + hitShaderCount;
    uint32_t handleStorageSize = handleCount * handleSize;
    std::vector<uint8_t> handleStorage(handleStorageSize);
    auto result = device->getRayTracingShaderGroupHandlesKHR(
        *pipeline, 0, handleCount, handleStorageSize, handleStorage.data());
    if (result != vk::Result::eSuccess) {
        std::cerr << "Failed to get ray tracing shader group handles.\n";
        std::abort();
    }
    // Copy handles
    uint8_t* sbtHead =
        static_cast<uint8_t*>(device->mapMemory(*sbt.memory, 0, sbtSize));

    uint8_t* dstPtr = sbtHead;
    auto copyHandle = [&](uint32_t index) {
        std::memcpy(dstPtr, handleStorage.data() + handleSize * index,
                    handleSize);
    };

    // Raygen
    uint32_t handleIndex = 0;
    copyHandle(handleIndex++);

    // Miss
    dstPtr = sbtHead + raygenRegion.size;
    for (uint32_t c = 0; c < missShaderCount; c++) {
        copyHandle(handleIndex++);
        dstPtr += missRegion.stride;
    }

    // Hit
    dstPtr = sbtHead + raygenRegion.size + missRegion.size;
    for (uint32_t c = 0; c < hitShaderCount; c++) {
        copyHandle(handleIndex++);
        dstPtr += hitRegion.stride;
    }

    raygenRegion.setDeviceAddress(sbt.address);
    missRegion.setDeviceAddress(sbt.address + raygenRegion.size);
    hitRegion.setDeviceAddress(sbt.address + raygenRegion.size +
                               missRegion.size);
}

/*
std::vector<Vertex> vertices = {
    Vertex{ glm::vec3(-0.5f, -0.5f, 0.0f ), glm::vec3( 0.0, 0.0, 1.0 ) },
    Vertex{ glm::vec3( 0.5f,  0.5f, 0.0f ), glm::vec3( 0.0, 1.0, 0.0 ) },
    Vertex{ glm::vec3(-0.5f,  0.5f, 0.0f ), glm::vec3( 1.0, 0.0, 0.0 ) },
    Vertex{ glm::vec3( 0.5f, -0.5f, 0.0f ), glm::vec3( 1.0, 1.0, 1.0 ) },
};

std::vector<uint32_t> indices = { 0, 1, 2, 1, 0, 3 };

SceneData sceneData = {
    glm::vec3{0.3f, -0.2f, 0.0f}
};
*/

void createTLAS(Object& object, SceneData scenedata, vk::PhysicalDevice physicalDevice,
                vk::Device device, vk::CommandPool commandPool,
                vk::Queue queue, AccelStruct& topAccel, AccelStruct& bottomAccel, Buffer& instanceBuffer) {
    // Create instance

    glm::mat4 matrix = object.getMatrix(0);
    glm::mat4 viewMatrix = scenedata.viewMatrix;
    
    glm::mat4 projectionMatrix = scenedata.projectionMatrix;
    projectionMatrix[1][1] *= -1; // Y軸反転

    matrix = projectionMatrix * viewMatrix * matrix;

    vk::TransformMatrixKHR transform = std::array{
        std::array{matrix[0][0], matrix[1][0], matrix[2][0], matrix[3][0]},
        std::array{matrix[0][1], matrix[1][1], matrix[2][1], matrix[3][1]},
        std::array{matrix[0][2], matrix[1][2], matrix[2][2], matrix[3][2]},
    };

    vk::AccelerationStructureInstanceKHR accelInstance{};
    accelInstance.setTransform(transform);
    accelInstance.setInstanceCustomIndex(0);
    accelInstance.setMask(0xFF);
    accelInstance.setInstanceShaderBindingTableRecordOffset(0);
    accelInstance.setFlags(
        vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable);
    accelInstance.setAccelerationStructureReference(
        bottomAccel.buffer.address);
    // Create buffer for instance
    instanceBuffer.init(
        physicalDevice, device,
        sizeof(vk::AccelerationStructureInstanceKHR),
        vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR |
        vk::BufferUsageFlagBits::eShaderDeviceAddress,
        vk::MemoryPropertyFlagBits::eHostVisible |
        vk::MemoryPropertyFlagBits::eHostCoherent,
        &accelInstance);
    // Create geometry
    vk::AccelerationStructureGeometryInstancesDataKHR instancesData{};
    instancesData.setArrayOfPointers(false);
    instancesData.setData(instanceBuffer.address);

    vk::AccelerationStructureGeometryKHR geometry{};
    geometry.setGeometryType(vk::GeometryTypeKHR::eInstances);
    geometry.setGeometry({instancesData});
    geometry.setFlags(vk::GeometryFlagBitsKHR::eOpaque);
    // Create and build TLAS
    constexpr uint32_t primitiveCount = 1;
    topAccel.init(physicalDevice, device, commandPool, queue,
                  vk::AccelerationStructureTypeKHR::eTopLevel,
                  geometry, primitiveCount);
}

void updateTLAS(Object& object, SceneData sceneData, vk::PhysicalDevice physicalDevice,
                vk::Device device, vk::CommandPool commandPool,
                vk::Queue queue, AccelStruct& topAccel, AccelStruct& bottomAccel,
                Buffer& instanceBuffer, uint32_t frameCount) {
    // フレームごとにインスタンスデータを更新
    glm::mat4 matrix = object.getMatrix(frameCount);
    glm::mat4 viewMatrix = sceneData.viewMatrix;
    
    glm::mat4 projectionMatrix = sceneData.projectionMatrix;
    projectionMatrix[1][1] *= -1; // Y軸反転

    matrix = projectionMatrix * viewMatrix * matrix;

    vk::TransformMatrixKHR transform = std::array{
        std::array{matrix[0][0], matrix[1][0], matrix[2][0], matrix[3][0]},
        std::array{matrix[0][1], matrix[1][1], matrix[2][1], matrix[3][1]},
        std::array{matrix[0][2], matrix[1][2], matrix[2][2], matrix[3][2]},
    };

    vk::AccelerationStructureInstanceKHR accelInstance{};
    accelInstance.setTransform(transform);
    accelInstance.setInstanceCustomIndex(0);
    accelInstance.setMask(0xFF);
    accelInstance.setInstanceShaderBindingTableRecordOffset(0);
    accelInstance.setFlags(
        vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable);
    accelInstance.setAccelerationStructureReference(
        bottomAccel.buffer.address);

    // インスタンスデータをバッファにコピー
    void* data;
    device.mapMemory(instanceBuffer.memory.get(), 0, sizeof(accelInstance), {}, &data);
    memcpy(data, &accelInstance, sizeof(accelInstance));
    device.unmapMemory(instanceBuffer.memory.get());

    // Create geometry
    vk::AccelerationStructureGeometryInstancesDataKHR instancesData{};
    instancesData.setArrayOfPointers(false);
    instancesData.setData(instanceBuffer.address);

    vk::AccelerationStructureGeometryKHR geometry{};
    geometry.setGeometryType(vk::GeometryTypeKHR::eInstances);
    geometry.setGeometry({instancesData});
    geometry.setFlags(vk::GeometryFlagBitsKHR::eOpaque);

    // TLASの再構築
    constexpr uint32_t primitiveCount = 1;
    topAccel.init(physicalDevice, device, commandPool, queue,
                  vk::AccelerationStructureTypeKHR::eTopLevel, geometry, primitiveCount);
}

inline vk::UniqueShaderModule createShaderModule(vk::Device device,
                                                 const std::string& ShaderPath) {
    size_t fileSz = std::filesystem::file_size(ShaderPath);
    std::ifstream spvFile(ShaderPath, std::ios_base::binary);
    std::vector<char> spvFileData(fileSz);
    spvFile.read(spvFileData.data(), fileSz);

    vk::ShaderModuleCreateInfo createInfo{};
    createInfo.setCodeSize(fileSz);
    createInfo.setPCode(reinterpret_cast<const uint32_t*>(spvFileData.data()));
    return device.createShaderModuleUnique(createInfo);
}

void addShader(uint32_t shaderIndex,
                const std::string& filename,
                vk::ShaderStageFlagBits stage, vk::Device device, 
                std::vector<vk::UniqueShaderModule>& shaderModules, 
                std::vector<vk::PipelineShaderStageCreateInfo>& shaderStages) {
    shaderModules[shaderIndex] =
        createShaderModule(device, filename);
    shaderStages[shaderIndex].setStage(stage);
    shaderStages[shaderIndex].setModule(*shaderModules[shaderIndex]);
    shaderStages[shaderIndex].setPName("main");
}

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
    }

    if (!existsSuitablePhysicalDevice) {
        std::cerr << "使用可能な物理デバイスがありません。" << std::endl;
        return -1;
    }

    //論理デバイスの作成

    // 機能の有効化
    vk::PhysicalDeviceBufferDeviceAddressFeaturesKHR bufferDeviceAddrFeatures;
    bufferDeviceAddrFeatures.bufferDeviceAddress = VK_TRUE;

    vk::PhysicalDeviceRayTracingPipelineFeaturesKHR rayTracingPipelineFeatures;
    rayTracingPipelineFeatures.rayTracingPipeline = VK_TRUE;
    rayTracingPipelineFeatures.pNext = &bufferDeviceAddrFeatures;

    vk::PhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructureFeatures;
    accelerationStructureFeatures.accelerationStructure = VK_TRUE;
    accelerationStructureFeatures.pNext = &rayTracingPipelineFeatures;

    vk::PhysicalDeviceDescriptorIndexingFeatures descriptorIndexingFeatures;
    descriptorIndexingFeatures.shaderUniformBufferArrayNonUniformIndexing = VK_TRUE;
    descriptorIndexingFeatures.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
    descriptorIndexingFeatures.runtimeDescriptorArray = VK_TRUE;
    descriptorIndexingFeatures.pNext = &accelerationStructureFeatures;

    // VkPhysicalDeviceFeatures2の設定
    vk::PhysicalDeviceFeatures features = physicalDevice.getFeatures();

    vk::PhysicalDeviceFeatures2 physicalDeviceFeatures2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, nullptr,};
    physicalDeviceFeatures2.features = features;
    physicalDeviceFeatures2.pNext = &descriptorIndexingFeatures;

    vk::DeviceCreateInfo devCreateInfo;

    auto devRequiredExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME, 

        // Vulkan Raytracing API で必要.
        VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
        VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,

        VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
        VK_KHR_SPIRV_1_4_EXTENSION_NAME,
        VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME,

        // descriptor indexing に必要.
        VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
        };
    
    devCreateInfo.enabledExtensionCount = devRequiredExtensions.size();
    devCreateInfo.ppEnabledExtensionNames = devRequiredExtensions.begin();
    devCreateInfo.enabledLayerCount = requiredLayers.size();
    devCreateInfo.ppEnabledLayerNames = requiredLayers.begin();
    devCreateInfo.pNext = &physicalDeviceFeatures2;
    
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
    std::vector<Object> objects = loadObjectsFromCSV("../../output.csv");

    uint32_t vertexCount = 0,indexCount = 0;;

    for(const auto& obj : objects) {
        vertexCount += obj.vertices.size();
        indexCount += obj.indices.size();
    }

    //カメラをCSVファイルから読み込む
    Camera camera = loadCameraFromCSV("../../camera.csv");

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
/*
    //頂点バッファの作成
    
    vk::BufferCreateInfo vertBufferCreateInfo;
    vertBufferCreateInfo.size = sizeof(Vertex) * vertexCount;
    vertBufferCreateInfo.usage = 
        vk::BufferUsageFlagBits::eVertexBuffer |
        vk::BufferUsageFlagBits::eTransferDst  |
        vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR |
        vk::BufferUsageFlagBits::eShaderDeviceAddress;

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
    indexBufferCreateInfo.size = sizeof(uint32_t) * indexCount;
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
        stagingBufferCreateInfo.size = sizeof(uint32_t) * indexCount;
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

        std::vector<uint32_t> allIndices;

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
        bufCopy.size = sizeof(uint32_t) * indexCount;

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
*/
    

    //ユニフォームバッファの作成

    SceneData sceneData = {
        //glm::mat4(1.0f),
        camera.viewMatrices.at(0),
        camera.projectionMatrices
    };

    // デバイスのプロパティを取得
    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);
    VkDeviceSize minUniformBufferOffsetAlignment = deviceProperties.limits.minUniformBufferOffsetAlignment;
        
    // シーンデータのサイズ
    VkDeviceSize sceneDataSize = sizeof(SceneData);

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
    
    //デスクリプタセットレイアウトの作成
    
    std::vector<vk::DescriptorSetLayoutBinding> descSetLayoutBinding(2);
    descSetLayoutBinding[0].binding = 0;
    descSetLayoutBinding[0].descriptorType = vk::DescriptorType::eAccelerationStructureKHR;
    descSetLayoutBinding[0].descriptorCount = 1;
    descSetLayoutBinding[0].stageFlags = vk::ShaderStageFlagBits::eRaygenKHR;

    descSetLayoutBinding[1].binding = 1;
    descSetLayoutBinding[1].descriptorType = vk::DescriptorType::eUniformBuffer;
    descSetLayoutBinding[1].descriptorCount = 1;
    descSetLayoutBinding[1].stageFlags = vk::ShaderStageFlagBits::eRaygenKHR;

    vk::DescriptorSetLayoutCreateInfo descSetLayoutCreateInfo{};
    descSetLayoutCreateInfo.bindingCount = static_cast<uint32_t>(descSetLayoutBinding.size());
    descSetLayoutCreateInfo.setBindings(descSetLayoutBinding);

    vk::UniqueDescriptorSetLayout descSetLayout = device->createDescriptorSetLayoutUnique(descSetLayoutCreateInfo);

    //デスクリプタプールの作成
    std::vector<vk::DescriptorPoolSize> descPoolSize = {
        {vk::DescriptorType::eAccelerationStructureKHR, 1},
        {vk::DescriptorType::eUniformBuffer, 1}
    };

    vk::DescriptorPoolCreateInfo descPoolCreateInfo;
    descPoolCreateInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
    descPoolCreateInfo.poolSizeCount = static_cast<uint32_t>(descPoolSize.size()); // poolSizeCountはdescPoolSizeの要素数
    descPoolCreateInfo.setPoolSizes(descPoolSize);
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
/*
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
*/
/*    //バーテックスシェーダーの読み込み

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
*/
    //パイプラインの作成

/*    vk::Viewport viewports[1];
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
/*
    vk::PipelineVertexInputStateCreateInfo vertexInputInfo;
    vertexInputInfo.vertexAttributeDescriptionCount = 4;
    vertexInputInfo.pVertexAttributeDescriptions = vertexInputDescription;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = vertexBindingDescription;
*/
/*    vk::PipelineInputAssemblyStateCreateInfo inputAssembly;
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
*/
    

    //シェーダーモジュールとシェーダーステージの設定

    std::vector<vk::UniqueShaderModule> shaderModules;
    std::vector<vk::PipelineShaderStageCreateInfo> shaderStages;
    std::vector<vk::RayTracingShaderGroupCreateInfoKHR> shaderGroups;

    uint32_t raygenShader = 0;
    uint32_t missShader = 1;
    uint32_t chitShader = 2;
    shaderStages.resize(3);
    shaderModules.resize(3);

    addShader(raygenShader, "raygen.rgen.spv",
              vk::ShaderStageFlagBits::eRaygenKHR, device.get(), shaderModules, shaderStages);
    addShader(missShader, "miss.rmiss.spv",
              vk::ShaderStageFlagBits::eMissKHR, device.get(), shaderModules, shaderStages);
    addShader(chitShader, "closesthit.rchit.spv",
              vk::ShaderStageFlagBits::eClosestHitKHR, device.get(), shaderModules, shaderStages);
    
    //シェーダーグループの設定
    uint32_t raygenGroup = 0;
    uint32_t missGroup = 1;
    uint32_t hitGroup = 2;
    shaderGroups.resize(3);

    // Raygen group
    shaderGroups[raygenGroup].setType(
        vk::RayTracingShaderGroupTypeKHR::eGeneral);
    shaderGroups[raygenGroup].setGeneralShader(raygenShader);
    shaderGroups[raygenGroup].setClosestHitShader(VK_SHADER_UNUSED_KHR);
    shaderGroups[raygenGroup].setAnyHitShader(VK_SHADER_UNUSED_KHR);
    shaderGroups[raygenGroup].setIntersectionShader(VK_SHADER_UNUSED_KHR);

    // Miss group
    shaderGroups[missGroup].setType(
        vk::RayTracingShaderGroupTypeKHR::eGeneral);
    shaderGroups[missGroup].setGeneralShader(missShader);
    shaderGroups[missGroup].setClosestHitShader(VK_SHADER_UNUSED_KHR);
    shaderGroups[missGroup].setAnyHitShader(VK_SHADER_UNUSED_KHR);
    shaderGroups[missGroup].setIntersectionShader(VK_SHADER_UNUSED_KHR);

    // Hit group
    shaderGroups[hitGroup].setType(
        vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup);
    shaderGroups[hitGroup].setGeneralShader(VK_SHADER_UNUSED_KHR);
    shaderGroups[hitGroup].setClosestHitShader(chitShader);
    shaderGroups[hitGroup].setAnyHitShader(VK_SHADER_UNUSED_KHR);
    shaderGroups[hitGroup].setIntersectionShader(VK_SHADER_UNUSED_KHR);

    //シェーダーバインディングテーブルの作成

    Buffer sbt{};
    vk::StridedDeviceAddressRegionKHR raygenRegion{};
    vk::StridedDeviceAddressRegionKHR missRegion{};
    vk::StridedDeviceAddressRegionKHR hitRegion{};


    //パイプラインレイアウトの作成
    auto pipelineDescSetLayouts = { descSetLayout.get() };

    vk::PipelineLayoutCreateInfo layoutCreateInfo;
    layoutCreateInfo.setSetLayouts(*descSetLayout);

    vk::UniquePipelineLayout pipelineLayout = device->createPipelineLayoutUnique(layoutCreateInfo);

    //レイトレーシングパイプラインの作成
    vk::RayTracingPipelineCreateInfoKHR pipelineCreateInfo{};
    pipelineCreateInfo.setLayout(*pipelineLayout);
    pipelineCreateInfo.setStages(shaderStages);
    pipelineCreateInfo.setGroups(shaderGroups);
    pipelineCreateInfo.setMaxPipelineRayRecursionDepth(1);
    auto resultPipe = device->createRayTracingPipelineKHRUnique(
        nullptr, nullptr, pipelineCreateInfo);
    if (resultPipe.result != vk::Result::eSuccess) {
        std::cerr << "Failed to create ray tracing pipeline.\n";
        std::abort();
    }
    auto pipeline = std::move(resultPipe.value);

/*    //シェーダーステージの設定
    vk::PipelineShaderStageCreateInfo shaderStage[2];
    shaderStage[0].stage = vk::ShaderStageFlagBits::eVertex;
    shaderStage[0].module = vertShader.get();
    shaderStage[0].pName = "main";
    shaderStage[1].stage = vk::ShaderStageFlagBits::eFragment;
    shaderStage[1].module = fragShader.get();
    shaderStage[1].pName = "main";

    vk::GraphicsPipelineCreateInfo pipelineCreateInfo;
    pipelineCreateInfo.pViewportState = &viewportState;
    //pipelineCreateInfo.pVertexInputState = &vertexInputInfo;
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
*/
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
        //frameBufCreateInfo.renderPass = renderpass.get();
        frameBufCreateInfo.attachmentCount = 1;
        frameBufCreateInfo.pAttachments = frameBufAttachments;

        swapchainFramebufs[i] = device->createFramebufferUnique(frameBufCreateInfo);
    }

    //コマンドプールの作成

    vk::CommandPoolCreateInfo cmdPoolCreateInfo;
    cmdPoolCreateInfo.queueFamilyIndex = graphicsQueueFamilyIndex;
    cmdPoolCreateInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;

    vk ::UniqueCommandPool cmdPool = device->createCommandPoolUnique(cmdPoolCreateInfo);

    //BLASの作成
    std::vector<AccelStruct> bottomLevelAS;
    for (auto& object : objects) {
        AccelStruct blas;
        createBLAS(object, physicalDevice, device.get(), cmdPool.get(), graphicsQueue, blas);
        bottomLevelAS.push_back(std::move(blas));
    }

    //TLASの作成
    std::vector<AccelStruct> topLevelAS;
    std::vector<Buffer> instanceBufs;

    for(int i = 0; i < objects.size(); i++) {
        AccelStruct tlas;
        Buffer instanceBuf;
        createTLAS(objects.at(i), sceneData, physicalDevice, device.get(), cmdPool.get(), graphicsQueue, tlas, bottomLevelAS.at(i), instanceBuf);
        topLevelAS.push_back(std::move(tlas));
        instanceBufs.push_back(std::move(instanceBuf));
    }

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

    while (!glfwWindowShouldClose(window) && frameCount < 50) {
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

        // TLASの更新
        for (int i = 0; i < objects.size(); i++) {
            updateTLAS(objects.at(i), sceneData, physicalDevice, device.get(), cmdPool.get(), graphicsQueue, topLevelAS.at(i), bottomLevelAS.at(i), instanceBufs.at(i), frameCount);
        }

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

        vk::ClearValue clearVal[1];
        clearVal[0].color.float32[0] = 0.0f;
        clearVal[0].color.float32[1] = 0.0f;
        clearVal[0].color.float32[2] = 0.0f;
        clearVal[0].color.float32[3] = 1.0f;

        vk::RenderPassBeginInfo renderpassBeginInfo;
        //renderpassBeginInfo.renderPass = renderpass.get();
        renderpassBeginInfo.framebuffer = swapchainFramebufs[imgIndex].get();
        renderpassBeginInfo.renderArea = vk::Rect2D({ 0,0 }, { screenWidth, screenHeight });
        renderpassBeginInfo.clearValueCount = 1;
        renderpassBeginInfo.pClearValues = clearVal;

        cmdBufs[0]->beginRenderPass(renderpassBeginInfo, vk::SubpassContents::eInline);

            // ここでサブパス0番の処理

        cmdBufs[0]->bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline.get());
        //cmdBufs[0]->bindVertexBuffers(0, { vertBuf.get() }, { 0 });
        //cmdBufs[0]->bindIndexBuffer(indexBuf.get(), 0, vk::IndexType::eUint16);
        cmdBufs[0]->bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout.get(), 0, { descSets[0].get() }, {});   //デスクリプタセットのバインド
        
        cmdBufs[0]->drawIndexed(indexCount, 1, 0, 0, 0);

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