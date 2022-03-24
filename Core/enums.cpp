#include "enums.h"

std::ostream& operator<<(std::ostream& out, const PixelFormat& pixelFormat)
{
    switch (pixelFormat)
    {
    case PixelFormat::Unknown:
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