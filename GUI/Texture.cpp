#include "Texture.h"
#include "MainWindow.h"
#ifdef _WIN32
#include <d3d11.h>
#endif // _WIN32

ACMB_GUI_NAMESPACE_BEGIN

Texture::Texture( std::shared_ptr<acmb::Bitmap<acmb::PixelFormat::RGBA32>> pBitmap )
: _width( pBitmap->GetWidth() )
, _height( pBitmap->GetHeight() )
{
    if ( !pBitmap )
        return;

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
    subResource.pSysMem = pBitmap->GetPlanarScanline( 0 );
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
#endif
}

Texture::~Texture()
{
#ifdef _WIN32
    if ( _pSRV )
        _pSRV->Release();
#endif
}

void* Texture::GetTexture() const
{
#ifdef _WIN32
    return _pSRV;
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
