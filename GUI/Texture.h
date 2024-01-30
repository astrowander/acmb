#pragma once

#include "../Core/bitmap.h"
#ifdef _WIN32
struct ID3D11ShaderResourceView;
#endif

ACMB_GUI_NAMESPACE_BEGIN

#ifdef __linux__
struct VulkanTextureData;
#endif

class Texture
{
#ifdef _WIN32
    ID3D11ShaderResourceView* _pSRV = nullptr;
#elif defined( __linux__ )
    std::shared_ptr<VulkanTextureData> _pTextureData = nullptr;
#endif // _WIN32

    uint32_t _width;
    uint32_t _height;

public:
    Texture( IBitmapPtr pBitmap );
    ~Texture();

    void* GetTexture() const;
    uint32_t GetWidth() const;
    uint32_t GetHeight() const;
};
ACMB_GUI_NAMESPACE_END
