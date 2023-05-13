#pragma once
#include "./../Core/macros.h"
#include <cstdint>

ACMB_NAMESPACE_BEGIN

enum class CommandCode : uint8_t
{
    Connect = 1,
    Disconnect = 2,

    SetDesiredFormat = 3,
    Binning = 4,
    Convert = 5,
    Subtract = 6,
    Divide = 7,
    AutoWB = 8,
    Deaberrate = 9,
    RemoveHalo = 10,
    Resize = 11,
    Crop = 12,
    Debayer = 13,

    Stack = 127
};

enum class ExtensionCode : uint8_t
{
    Ppm,
    Tiff,
    Jpeg
};

ACMB_NAMESPACE_END
