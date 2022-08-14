#ifndef IMAGEDECODER_H
#define IMAGEDECODER_H

#include <string>
#include <vector>
#include <iostream>
#include <memory>
#include "../Core/imageparams.h"
#include "../Core/camerasettings.h"
#include <sstream>
#include <unordered_set>

class IBitmap;

class ImageDecoder : public ImageParams
{
protected:
    std::string _lastFileName;
    std::shared_ptr<std::istream> _pStream;
    std::shared_ptr<CameraSettings> _pCameraSettings = std::make_shared<CameraSettings>();
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

    static std::shared_ptr<ImageDecoder> Create(const std::string& fileName);

    const std::string& GetLastFileName()
    {
        return _lastFileName;
    }

    std::shared_ptr<CameraSettings> GetCameraSettings()
    {
        return _pCameraSettings;
    }

    static std::vector<std::shared_ptr<ImageDecoder>> GetDecodersFromDir( std::string path );
    static std::vector<std::shared_ptr<ImageDecoder>> GetDecodersFromMask( std::string mask );

    static const std::unordered_set<std::string>& GetAllExtensions()
    {
        return _allExtensions;
    }

protected:
    static bool AddCommonExtensions( const std::unordered_set<std::string>& extensions )
    {
        _allExtensions.insert( std::begin(extensions), std::end(extensions) );
        return true;
    }
};

#define ADD_EXTENSIONS inline static bool handle = AddCommonExtensions(GetExtensions());
#endif // IMAGEDECODER_H
