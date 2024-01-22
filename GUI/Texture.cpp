#include "Texture.h"
#include "MainWindow.h"

#include "./../Transforms/converter.h"
#ifdef _WIN32
#include <d3d11.h>
#elif defined ( __linux__ )
#include "imgui_impl_vulkan.h"
#endif // _WIN32

ACMB_GUI_NAMESPACE_BEGIN

#ifdef __linux__

static bool check_vk_result(VkResult err)
{
    if (err == 0)
        return true;
    fprintf(stderr, "[vulkan] Error: VkResult = %d\n", err);
    return false;
}

struct VulkanTextureData
{
    VkDescriptorSet DS;         // Descriptor set: this is what you'll pass to Image()
    int             Width;
    int             Height;
    int             Channels;

    // Need to keep track of these to properly cleanup
    VkImageView     ImageView;
    VkImage         Image;
    VkDeviceMemory  ImageMemory;
    VkSampler       Sampler;
    VkBuffer        UploadBuffer;
    VkDeviceMemory  UploadBufferMemory;

    VulkanTextureData() { memset(this, 0, sizeof(*this)); }
};

// Helper function to find Vulkan memory type bits. See ImGui_ImplVulkan_MemoryType() in imgui_impl_vulkan.cpp
uint32_t findMemoryType(uint32_t type_filter, VkMemoryPropertyFlags properties)
{
    VkPhysicalDeviceMemoryProperties mem_properties;
    auto& mainWindow = MainWindow::GetInstance( FontRegistry::Instance() );
    vkGetPhysicalDeviceMemoryProperties( mainWindow.GetPhysicalDevice() , &mem_properties );

    for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++)
        if ((type_filter & (1 << i)) && (mem_properties.memoryTypes[i].propertyFlags & properties) == properties)
            return i;

    return 0xFFFFFFFF; // Unable to find memoryType
}

// Helper function to load an image with common settings and return a MyTextureData with a VkDescriptorSet as a sort of Vulkan pointer
bool LoadTextureFromMemory(unsigned char* image_data, VulkanTextureData* tex_data)
{
    if (image_data == NULL || tex_data == NULL )
        return false;

    // Calculate allocation size (in number of bytes)
    size_t image_size = tex_data->Width * tex_data->Height * tex_data->Channels;
    VkResult err;
    auto& mainWindow = MainWindow::GetInstance( FontRegistry::Instance() );
    auto g_Device = mainWindow.GetDevice();
    auto g_Allocator = mainWindow.GetAllocator();
    auto g_MainWindowData = *mainWindow.GetMainWindowData();
    auto g_Queue = mainWindow.GetQueue();
    // Create the Vulkan image.
    {
        VkImageCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        info.imageType = VK_IMAGE_TYPE_2D;
        info.format = VK_FORMAT_R8G8B8A8_UNORM;
        info.extent.width = tex_data->Width;
        info.extent.height = tex_data->Height;
        info.extent.depth = 1;
        info.mipLevels = 1;
        info.arrayLayers = 1;
        info.samples = VK_SAMPLE_COUNT_1_BIT;
        info.tiling = VK_IMAGE_TILING_OPTIMAL;
        info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        err = vkCreateImage(g_Device, &info, g_Allocator, &tex_data->Image);
        if ( !check_vk_result(err) ) return false;
        VkMemoryRequirements req;
        vkGetImageMemoryRequirements(g_Device, tex_data->Image, &req);
        VkMemoryAllocateInfo alloc_info = {};
        alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc_info.allocationSize = req.size;
        alloc_info.memoryTypeIndex = findMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        err = vkAllocateMemory(g_Device, &alloc_info, g_Allocator, &tex_data->ImageMemory);
        if ( !check_vk_result(err) ) return false;
        err = vkBindImageMemory(g_Device, tex_data->Image, tex_data->ImageMemory, 0);
        if ( !check_vk_result(err) ) return false;
    }

    // Create the Image View
    {
        VkImageViewCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        info.image = tex_data->Image;
        info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        info.format = VK_FORMAT_R8G8B8A8_UNORM;
        info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        info.subresourceRange.levelCount = 1;
        info.subresourceRange.layerCount = 1;
        err = vkCreateImageView(g_Device, &info, g_Allocator, &tex_data->ImageView);
        if ( !check_vk_result(err) ) return false;
    }

    // Create Sampler
    {
        VkSamplerCreateInfo sampler_info{};
        sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sampler_info.magFilter = VK_FILTER_LINEAR;
        sampler_info.minFilter = VK_FILTER_LINEAR;
        sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT; // outside image bounds just use border color
        sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sampler_info.minLod = -1000;
        sampler_info.maxLod = 1000;
        sampler_info.maxAnisotropy = 1.0f;
        err = vkCreateSampler(g_Device, &sampler_info, g_Allocator, &tex_data->Sampler);
        if ( !check_vk_result(err) ) return false;
    }

    // Create Descriptor Set using ImGUI's implementation
    tex_data->DS = ImGui_ImplVulkan_AddTexture(tex_data->Sampler, tex_data->ImageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    // Create Upload Buffer
    {
        VkBufferCreateInfo buffer_info = {};
        buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buffer_info.size = image_size;
        buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        err = vkCreateBuffer(g_Device, &buffer_info, g_Allocator, &tex_data->UploadBuffer);
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(g_Device, tex_data->UploadBuffer, &req);
        VkMemoryAllocateInfo alloc_info = {};
        alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc_info.allocationSize = req.size;
        alloc_info.memoryTypeIndex = findMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
        err = vkAllocateMemory(g_Device, &alloc_info, g_Allocator, &tex_data->UploadBufferMemory);
        if ( !check_vk_result(err) ) return false;
        err = vkBindBufferMemory(g_Device, tex_data->UploadBuffer, tex_data->UploadBufferMemory, 0);
        if ( !check_vk_result(err) ) return false;
    }

    // Upload to Buffer:
    {
        void* map = NULL;
        err = vkMapMemory(g_Device, tex_data->UploadBufferMemory, 0, image_size, 0, &map);
        memcpy(map, image_data, image_size);
        VkMappedMemoryRange range[1] = {};
        range[0].sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        range[0].memory = tex_data->UploadBufferMemory;
        range[0].size = image_size;
        err = vkFlushMappedMemoryRanges(g_Device, 1, range);
        if ( !check_vk_result(err) ) return false;
        vkUnmapMemory(g_Device, tex_data->UploadBufferMemory);
    }

    // Create a command buffer that will perform following steps when hit in the command queue.
    // TODO: this works in the example, but may need input if this is an acceptable way to access the pool/create the command buffer.
    VkCommandPool command_pool = g_MainWindowData.Frames[g_MainWindowData.FrameIndex].CommandPool;
    VkCommandBuffer command_buffer;
    {
        VkCommandBufferAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_info.commandPool = command_pool;
        alloc_info.commandBufferCount = 1;

        err = vkAllocateCommandBuffers(g_Device, &alloc_info, &command_buffer);
        if ( !check_vk_result(err) ) return false;
        VkCommandBufferBeginInfo begin_info = {};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        err = vkBeginCommandBuffer(command_buffer, &begin_info);
        if ( !check_vk_result(err) ) return false;
    }

    // Copy to Image
    {
        VkImageMemoryBarrier copy_barrier[1] = {};
        copy_barrier[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        copy_barrier[0].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        copy_barrier[0].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        copy_barrier[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        copy_barrier[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        copy_barrier[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        copy_barrier[0].image = tex_data->Image;
        copy_barrier[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copy_barrier[0].subresourceRange.levelCount = 1;
        copy_barrier[0].subresourceRange.layerCount = 1;
        vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, copy_barrier);

        VkBufferImageCopy region = {};
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.layerCount = 1;
        region.imageExtent.width = tex_data->Width;
        region.imageExtent.height = tex_data->Height;
        region.imageExtent.depth = 1;
        vkCmdCopyBufferToImage(command_buffer, tex_data->UploadBuffer, tex_data->Image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        VkImageMemoryBarrier use_barrier[1] = {};
        use_barrier[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        use_barrier[0].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        use_barrier[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        use_barrier[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        use_barrier[0].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        use_barrier[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        use_barrier[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        use_barrier[0].image = tex_data->Image;
        use_barrier[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        use_barrier[0].subresourceRange.levelCount = 1;
        use_barrier[0].subresourceRange.layerCount = 1;
        vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, NULL, 0, NULL, 1, use_barrier);
    }

    // End command buffer
    {
        VkSubmitInfo end_info = {};
        end_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        end_info.commandBufferCount = 1;
        end_info.pCommandBuffers = &command_buffer;
        err = vkEndCommandBuffer(command_buffer);
        if ( !check_vk_result(err) ) return false;
        err = vkQueueSubmit(g_Queue, 1, &end_info, VK_NULL_HANDLE);
        if ( !check_vk_result(err) ) return false;
        err = vkDeviceWaitIdle(g_Device);
        if ( !check_vk_result(err) ) return false;
    }

    return true;
}

// Helper function to cleanup an image loaded with LoadTextureFromFile
void RemoveTexture(VulkanTextureData* tex_data)
{
    if ( !tex_data )
        return;

    auto& mainWindow = MainWindow::GetInstance( FontRegistry::Instance() );
    auto g_Device = mainWindow.GetDevice();
    vkFreeMemory(g_Device, tex_data->UploadBufferMemory, nullptr);
    vkDestroyBuffer(g_Device, tex_data->UploadBuffer, nullptr);
    vkDestroySampler(g_Device, tex_data->Sampler, nullptr);
    vkDestroyImageView(g_Device, tex_data->ImageView, nullptr);
    vkDestroyImage(g_Device, tex_data->Image, nullptr);
    ImGui_ImplVulkan_RemoveTexture(tex_data->DS);
}

#endif

Texture::Texture( IBitmapPtr pBitmap )
: _width( pBitmap->GetWidth() )
, _height( pBitmap->GetHeight() )
{
    if ( !pBitmap )
        return;

    auto pTextureBitmap = Converter::Convert( pBitmap, PixelFormat::RGBA32 );
#ifdef _WIN32
    // Create texture
    D3D11_TEXTURE2D_DESC desc;
    ZeroMemory( &desc, sizeof( desc ) );
    desc.Width = _width;
    desc.Height = _height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    desc.CPUAccessFlags = 0;

    ID3D11Texture2D* pTexture = NULL;
    D3D11_SUBRESOURCE_DATA subResource;
    subResource.pSysMem = pTextureBitmap->GetPlanarScanline( 0 );
    subResource.SysMemPitch = desc.Width * 4;
    subResource.SysMemSlicePitch = 0;
    auto pd3dDevice = acmb::gui::MainWindow::GetInstance( acmb::gui::FontRegistry::Instance() ).GetD3D11Device();
    pd3dDevice->CreateTexture2D( &desc, &subResource, &pTexture );
    if ( !pTexture )
        return;

    // Create texture view
    D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc;
    ZeroMemory( &srvDesc, sizeof( srvDesc ) );
    srvDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Texture2D.MipLevels = desc.MipLevels;
    srvDesc.Texture2D.MostDetailedMip = 0;
    pd3dDevice->CreateShaderResourceView( pTexture, &srvDesc, &_pSRV );
    pTexture->Release();
#elif defined ( __linux__ )
    _pTextureData = std::make_shared<VulkanTextureData>();
    _pTextureData->Width = pBitmap->GetWidth();
    _pTextureData->Height = pBitmap->GetHeight();
    _pTextureData->Channels = 4;
    LoadTextureFromMemory( pTextureBitmap->GetData().data(), _pTextureData.get() );
#endif
}

Texture::~Texture()
{
#ifdef _WIN32
    if ( _pSRV )
        _pSRV->Release();
#elif defined ( __linux__ )
    RemoveTexture(_pTextureData.get() );
    _pTextureData.reset();
#endif

}

void* Texture::GetTexture() const
{
#ifdef _WIN32
    return _pSRV;
#elif defined ( __linux__ )
    return _pTextureData->DS;
#endif
}

uint32_t Texture::GetWidth() const
{
    return _width;
}

uint32_t Texture::GetHeight() const
{
    return _height;
}

ACMB_GUI_NAMESPACE_END
