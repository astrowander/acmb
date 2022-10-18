#pragma once
#include "../../Core/enums.h"

ACMB_NAMESPACE_BEGIN

/// Settings for files reading
struct DecoderSettings
{
    /// if true bitmap size will be two times smaller
    bool halfSize = false;
    /// if format is specified and not the same as decoded one picture will be converted
    PixelFormat outputFormat = PixelFormat::Unspecified;
};

ACMB_NAMESPACE_END
