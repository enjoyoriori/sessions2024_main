#include <iostream>
#include <vector>
#include <fstream>
#include <filesystem>
#include <cstring>
#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>

// cmake .. -DCMAKE_TOOLCHAIN_FILE=C:\vcpkg/scripts/buildsystems/vcpkg.cmake
// cmake --build .

const uint32_t screenWidth = 1920;
const uint32_t screenHeight = 1080;

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

struct Vec2 {
    float x, y;
};

struct Vec3 {
    float x, y, z;
};

struct Vertex {
    Vec3 pos;
    Vec3 color;
    Vec3 normal;
};

std::vector<Vertex> vertices = {
    Vertex{ Vec3{  0.0f, -0.5f, 0.0f } },
    Vertex{ Vec3{  0.5f,  0.5f, 0.0f } },
    Vertex{ Vec3{ -0.5f,  0.5f, 0.0f } }
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

    //頂点バッファの作成
    vk::BufferCreateInfo vertBufferCreateInfo;
    vertBufferCreateInfo.size = sizeof(Vertex) * vertices.size();
    vertBufferCreateInfo.usage = vk::BufferUsageFlagBits::eVertexBuffer;
    vertBufferCreateInfo.sharingMode = vk::SharingMode::eExclusive;

    vk::UniqueBuffer vertexBuf = device->createBufferUnique(vertBufferCreateInfo);

    //バッファのメモリ割り当て
    vk::PhysicalDeviceMemoryProperties memProps = physicalDevice.getMemoryProperties();

    vk::MemoryRequirements vertexBufMemReq = device->getBufferMemoryRequirements(vertexBuf.get());

    vk::MemoryAllocateInfo vertexBufMemAllocInfo;
    vertexBufMemAllocInfo.allocationSize = vertexBufMemReq.size;

    if (!memoryChecker(memProps, vertexBufMemReq, vertexBufMemAllocInfo, vk::MemoryPropertyFlagBits::eHostVisible)) {
        return -1;
    }

    vk::UniqueDeviceMemory vertexBufMemory = device->allocateMemoryUnique(vertexBufMemAllocInfo);

    device->bindBufferMemory(vertexBuf.get(), vertexBufMemory.get(), 0);

    //メモリマッピング

    void* vertexBufMem = device->mapMemory(vertexBufMemory.get(), 0, vertexBufMemReq.size);         //sizeof(Vertex) * vertices.size());でやったらエラーでた

    std::memcpy(vertexBufMem, vertices.data(), vertexBufMemReq.size);

    vk::MappedMemoryRange flushMemoryRange;
    flushMemoryRange.memory = vertexBufMemory.get();
    flushMemoryRange.offset = 0;
    flushMemoryRange.size = vertexBufMemReq.size;

    device->flushMappedMemoryRanges({ flushMemoryRange });

    device->unmapMemory(vertexBufMemory.get());

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

    vk::PipelineLayoutCreateInfo layoutCreateInfo;
    layoutCreateInfo.setLayoutCount = 0;
    layoutCreateInfo.pSetLayouts = nullptr;

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

    while (!glfwWindowShouldClose(window)) {
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

        //std::cout << "imgIndex: " << imgIndex << std::endl;

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
        cmdBufs[0]->bindVertexBuffers(0, { vertexBuf.get() }, { 0 });
        cmdBufs[0]->draw(vertices.size(), 1, 0, 0);

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
    }

    graphicsQueue.waitIdle();
    glfwTerminate();
    return 0;
}