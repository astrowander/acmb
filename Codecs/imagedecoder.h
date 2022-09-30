#pragma once

#include "../Core/macros.h"
#include "../Core/imageparams.h"
#include "../Core/camerasettings.h"
#include "../Core/pipeline.h"

#include <string>
#include <vector>
#include <iostream>
#include <memory>
#include <sstream>
#include <unordered_set>

ACMB_NAMESPACE_BEGIN

class IBitmap;

class ImageDecoder : public IPipelineFirstElement
{
protected:
    std::string _lastFileName;
    std::shared_ptr<std::istream> _pStream;    
    virtual std::unique_ptr<std::istringstream> ReadLine();

    inline static std::unordered_set<std::string> _allExtensions;

public:

    virtual void Attach(std::shared_ptr<std::istream> pStream);
    virtual void Attach(const std::string& fileName);
    virtual void Reattach();
    virtual void Detach();
    virtual ~ImageDecoder() = default;

    virtual std::shared_ptr<IBitmap> ReadBitmap() = 0;
    virtual std::shared_ptr<IBitmap> ReadStripe(uint32_t stripeHeight) = 0;
    virtual uint32_t GetCurrentScanline() const = 0;

    virtual IBitmapPtr ProcessBitmap( IBitmapPtr pBitmap = nullptr ) override;

    static std::shared_ptr<ImageDecoder> Create(const std::string& fileName);

    const std::string& GetLastFileName() const;

    static std::vector<Pipeline> GetPipelinesFromDir( std::string path );
    static std::vector<Pipeline> GetPipelinesFromMask( std::string mask );

    static const std::unordered_set<std::string>& GetAllExtensions();

protected:
    static bool AddCommonExtensions( const std::unordered_set<std::string>& extensions );
};

#define ADD_EXTENSIONS inline static bool handle = AddCommonExtensions(GetExtensions());

ACMB_NAMESPACE_END
