#pragma once
#include "../Core/macros.h"
#include "../Core/IPipelineElement.h"
#include <string>
#include <ostream>
#include <memory>
#include <unordered_set>

ACMB_NAMESPACE_BEGIN

class IBitmap;

class ImageEncoder: public IPipelineElement
{
protected:
    std::shared_ptr<std::ostream> _pStream;
    inline static std::unordered_set<std::string> _allExtensions;

public:

    virtual void Attach(std::shared_ptr<std::ostream> pStream);
    virtual void Attach(const std::string& fileName);
    virtual void Detach();

    virtual ~ImageEncoder() = default;

    virtual void WriteBitmap(std::shared_ptr<IBitmap> pBitmap) = 0;

    static std::shared_ptr<ImageEncoder> Create(const std::string& fileName);

    static const std::unordered_set<std::string>& GetAllExtensions();

    virtual IBitmapPtr ProcessBitmap( IBitmapPtr pBitmap ) override;

protected:
    static bool AddCommonExtensions( const std::unordered_set<std::string>& extensions );
};

#define ADD_EXTENSIONS inline static bool handle = AddCommonExtensions(GetExtensions());

ACMB_NAMESPACE_END
