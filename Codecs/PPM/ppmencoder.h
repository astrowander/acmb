#pragma once

#include "../../Core/enums.h"
#include "../../Codecs/imageencoder.h"

ACMB_NAMESPACE_BEGIN
/// <summary>
/// Writes bitmaps in the PPM format
/// </summary>
class PpmEncoder : public ImageEncoder
{
    PpmMode _ppmMode;
public:
    
    /// <param name="ppmMode">text or binary mode</param>
    PpmEncoder(PpmMode ppmMode);
    /// writes given bitmap
    void WriteBitmap(std::shared_ptr<IBitmap> pBitmap) override;
    /// returns supported extensions
    static std::set<std::string> GetExtensions();
private:
    template<uint32_t bytes>
    void WriteBinary(std::shared_ptr<IBitmap> pBitmap);

    void WriteText(std::shared_ptr<IBitmap> pBitmap);

    ADD_EXTENSIONS
};

ACMB_NAMESPACE_END
