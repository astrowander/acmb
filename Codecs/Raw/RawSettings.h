#pragma once
#include "../../Core/enums.h"

ACMB_NAMESPACE_BEGIN

/// Settings for RAW files reading
struct RawSettings
{
    /// if true bitmap size will be two times smaller
    bool halfSize = false;
    /// if format is grayscale picture will not be debayered
    PixelFormat outputFormat = PixelFormat::Gray16;
};

ACMB_NAMESPACE_END
