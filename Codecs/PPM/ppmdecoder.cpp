#include "ppmdecoder.h"
#include "Core/bitmap.h"

void PpmDecoder::Attach(const std::string &fileName)
{
    ImageDecoder::Attach(fileName);
}

void PpmDecoder::Attach(std::shared_ptr<std::istream> pStream)
{
    ImageDecoder::Attach(std::move(pStream));

    auto pStringStream = ReadLine();
    char magicNumber[2];
    pStringStream->read(magicNumber, 2);

    if (magicNumber[0] != 'P')
        throw std::runtime_error("file is corrupted");

    pStringStream = ReadLine();

    *pStringStream >> _width;
    *pStringStream >> _height;

    pStringStream = ReadLine();
    *pStringStream >> _maxval;

    if (_maxval > 65535)
        throw std::runtime_error("file is corrupted");

    switch (magicNumber[1])
    {
    case '2':
        _ppmMode = PpmMode::Text;
        _pixelFormat = (_maxval < 256) ? PixelFormat::Gray8 : PixelFormat::Gray16;
        break;
    case '3':
        _ppmMode = PpmMode::Text;
        _pixelFormat = (_maxval < 256) ? PixelFormat::RGB24 : PixelFormat::RGB48;
        break;
    case '5':
        _ppmMode = PpmMode::Binary;
        _pixelFormat = (_maxval < 256) ? PixelFormat::Gray8 : PixelFormat::Gray16;
        break;
    case '6':
        _ppmMode = PpmMode::Binary;
        _pixelFormat = (_maxval < 256) ? PixelFormat::RGB24 : PixelFormat::RGB48;
        break;
    default:
        throw std::runtime_error("not supported");
    }
}

template<>
void PpmDecoder::ParseBinary<1>()
{
    for (uint32_t i = 0; i < _height; ++i)
    {
        auto pScanline = _pBitmap->GetPlanarScanline(i);
        _pStream->read(pScanline, _width * BytesPerPixel(_pixelFormat));
    }
}

template<>
void PpmDecoder::ParseBinary<2>()
{
    for (uint32_t i = 0; i < _height; ++i)
    {
        auto pScanline = _pBitmap->GetPlanarScanline(i);
        for (uint32_t j = 0; j < _width * ChannelCount(_pixelFormat); ++j)
        {
            char bytes[2];
            _pStream->read(bytes, 2);
            *pScanline++ = bytes[1];
            *pScanline++ = bytes[0];
        }
    }
}

std::shared_ptr<IBitmap> PpmDecoder::GetBitmap()
{
    if (_pBitmap)
        return _pBitmap;

    if (!_pStream)
        throw std::runtime_error("decoder is detached");

    switch(_pixelFormat)
    {
    case PixelFormat::Gray8:
        _pBitmap.reset(new Bitmap<PixelFormat::Gray8>(_width, _height));
        break;
    case PixelFormat::Gray16:
        _pBitmap.reset(new Bitmap<PixelFormat::Gray16>(_width, _height));
        break;
    case PixelFormat::RGB24:
        _pBitmap.reset(new Bitmap<PixelFormat::RGB24>(_width, _height));
        break;
    case PixelFormat::RGB48:
        _pBitmap.reset(new Bitmap<PixelFormat::RGB48>(_width, _height));
        break;
    default:
        throw std::runtime_error("not implemented");
    }

    if (_ppmMode == PpmMode::Text)
        ParseText();
    else
        BytesPerChannel(_pixelFormat) == 1 ? ParseBinary<1>() : ParseBinary<2>();

    return _pBitmap;
}

void PpmDecoder::ParseText()
{
    for (uint32_t i = 0; i < _height; ++i)
    {
        for (uint32_t j = 0; j < _width; ++j)
        {
            for (uint32_t k = 0; k < ChannelCount(_pixelFormat); ++k)
            {
                uint32_t v;
                *_pStream >> v;
                _pBitmap->SetChannel(i, j, k, v);
            }
        }
    }
}
