#include "enums.h"
ACMB_NAMESPACE_BEGIN
std::ostream& operator<<(std::ostream& out, const PixelFormat& pixelFormat)
{
    switch (pixelFormat)
    {
    case PixelFormat::Unspecified:
        return out << "Unknown";
    case PixelFormat::Gray8:
        return out << "Gray8";
    case PixelFormat::Gray16:
        return out << "Gray16";
    case PixelFormat::RGB24:
        return out << "RGB24";
    case PixelFormat::RGB48:
        return out << "RGB248";
    }

    throw std::runtime_error("Unexpexted pixel format");
}

PixelFormat ConstructPixelFormat( uint16_t bitsPerChannel, uint16_t channelsPerPixel )
{
    switch ( channelsPerPixel )
    {
        case 1:
            return ( bitsPerChannel == 8 ) ? PixelFormat::Gray8 : ( bitsPerChannel == 16 ) ? PixelFormat::Gray16 : PixelFormat::Unspecified;
        case 3:
            return ( bitsPerChannel == 8 ) ? PixelFormat::RGB24 : ( bitsPerChannel == 16 ) ? PixelFormat::RGB48 : PixelFormat::Unspecified;
        default:
            return PixelFormat::Unspecified;
    }
}

ACMB_NAMESPACE_END