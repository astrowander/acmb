#pragma once

#include "../Core/bitmap.h"
#ifdef _WIN32
struct ID3D11ShaderResourceView;
#endif

ACMB_GUI_NAMESPACE_BEGIN

class Texture
{
#ifdef _WIN32
    ID3D11ShaderResourceView* _pSRV = nullptr;
#endif // _WIN32

    uint32_t _width;
    uint32_t _height;

public:
    Texture( std::shared_ptr<acmb::Bitmap<acmb::PixelFormat::RGBA32>> pBitmap );
    ~Texture();

    void* GetTexture() const;
    uint32_t GetWidth() const;
    uint32_t GetHeight() const;
};
ACMB_GUI_NAMESPACE_END