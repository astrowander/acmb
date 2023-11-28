#include "FitsDecoder.h"
#include <CCfits/CCfits>

ACMB_NAMESPACE_BEGIN

FitsDecoder::FitsDecoder( PixelFormat outputFormat )
: ImageDecoder( outputFormat )
{
}

void FitsDecoder::Attach( const std::string& fileName )
{
    _lastFileName = fileName;
    _pFits = new CCfits::FITS( fileName, CCfits::RWmode::Read, true );
    const auto bitpix = _pFits->pHDU().bitpix();

    switch ( bitpix )
    {
        case 8:
        case 16:
        default:
            throw std::runtime_error( "unsupported FITS format" );
    }
}

ACMB_NAMESPACE_END
