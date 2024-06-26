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
        return out << "RGB48";
    case PixelFormat::Bayer16:
        return out << "Bayer16";
    case PixelFormat::RGBA32:
        return out << "RGBA32";
    case PixelFormat::RGBA64:
        return out << "RGBA64";
    }

    throw std::runtime_error("Unexpexted pixel format");
}

ACMB_NAMESPACE_END
