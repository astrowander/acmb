#pragma once
#include "../Core/macros.h"
#include "../Core/IPipelineElement.h"
#include <string>
#include <ostream>
#include <memory>
#include <unordered_set>

ACMB_NAMESPACE_BEGIN

class IBitmap;
/// <summary>
/// Abstract class for writing bitmap to a file or a stream
/// </summary>
class ImageEncoder: public IPipelineElement
{
protected:
    std::shared_ptr<std::ostream> _pStream;
    inline static std::unordered_set<std::string> _allExtensions;

public:
    /// attach decoder to a stream
    virtual void Attach(std::shared_ptr<std::ostream> pStream);
    /// attach decoder to a file
    virtual void Attach(const std::string& fileName);
    /// detach decoder
    virtual void Detach();

    virtual ~ImageEncoder() = default;
    /// write given bitmap
    virtual void WriteBitmap(std::shared_ptr<IBitmap> pBitmap) = 0;
    /// needed for the compatibility with pipelines
    static std::shared_ptr<ImageEncoder> Create(const std::string& fileName);
    /// returns all supported extensions by all encoders
    static const std::unordered_set<std::string>& GetAllExtensions();
    /// needed for the compatibility with pipelines
    virtual IBitmapPtr ProcessBitmap( IBitmapPtr pBitmap ) override;

protected:
    static bool AddCommonExtensions( const std::unordered_set<std::string>& extensions );
};

#define ADD_EXTENSIONS inline static bool handle = AddCommonExtensions(GetExtensions());

ACMB_NAMESPACE_END
