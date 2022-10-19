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
/// <summary>
/// Abstract class for reading bitmap from a file or a stream
/// </summary>
class ImageDecoder : public IPipelineFirstElement
{
protected:
    std::string _lastFileName;
    std::shared_ptr<std::istream> _pStream;    
    virtual std::unique_ptr<std::istringstream> ReadLine();

    inline static std::unordered_set<std::string> _allExtensions;
    
    PixelFormat _decodedFormat = PixelFormat::Unspecified;

    ImageDecoder( PixelFormat outputFormat );

public:
    /// attach decoder to stream
    virtual void Attach(std::shared_ptr<std::istream> pStream);
    /// attach decoder to file
    virtual void Attach(const std::string& fileName);
    /// detach and attach again to read file from beginning
    virtual void Reattach();
    /// detach decoder
    virtual void Detach();
    virtual ~ImageDecoder() = default;
    
    /// read whole bitmap, need to implement in the derived class
    virtual std::shared_ptr<IBitmap> ReadBitmap() = 0;
    /// read a stripe (several lines), throws exception "not implemented, override this if the image format allows to read file partially.
    virtual std::shared_ptr<IBitmap> ReadStripe( uint32_t stripeHeight );
    /// returns beginning of the next stripe, throws exception "not implemented, override this if the image format allows to read file partially.
    virtual uint32_t GetCurrentScanline() const;
    /// reads and returns bitmap, needed for the compatibility with pipelines
    virtual IBitmapPtr ProcessBitmap( IBitmapPtr pBitmap = nullptr ) override;
    /// needed for the compatibility with pipelines, if opening file is RAW, rawSettings will be applied
    static std::shared_ptr<ImageDecoder> Create( const std::string& fileName, PixelFormat outputFormat = PixelFormat::Unspecified );
    /// returns name of the last attached file, if no file was attached returns empty string
    const std::string& GetLastFileName() const;

    /// finds all files of supported formats in given directory, attaches decoders to them and creates pipelines
    /// if opening file is RAW, rawSettings will be applied
    static std::vector<Pipeline> GetPipelinesFromDir( std::string path, PixelFormat outputFormat = PixelFormat::Unspecified );
    /// finds all files of supported formats satisfying given mask, attaches decoders to them and creates pipelines
    /// if opening file is RAW, rawSettings will be applied
    static std::vector<Pipeline> GetPipelinesFromMask( std::string mask, PixelFormat outputFormat = PixelFormat::Unspecified );
    /// returns all supported extensions by all decoders
    static const std::unordered_set<std::string>& GetAllExtensions();    

protected:
    static bool AddCommonExtensions( const std::unordered_set<std::string>& extensions );
    
    IBitmapPtr ToOutputFormat( IBitmapPtr pSrcBitmap );
};

#define ADD_EXTENSIONS inline static bool handle = AddCommonExtensions(GetExtensions());

ACMB_NAMESPACE_END
