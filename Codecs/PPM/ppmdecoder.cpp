#include "ppmdecoder.h"
#include "../../Core/bitmap.h"

void PpmDecoder::Attach(const std::string &fileName)
{
    ImageDecoder::Attach(fileName);
}

void PpmDecoder::Attach(std::shared_ptr<std::istream> pStream)
{
    ImageDecoder::Attach(pStream);

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

    _dataOffset = pStream->tellg();
}

template<>
std::shared_ptr<IBitmap> PpmDecoder::ReadBinaryStripe<1>(uint32_t stripeHeight)
{
    auto res = CreateStripe(stripeHeight);

    auto pScanline = res->GetPlanarScanline(0);
    _pStream->read(pScanline, _width * stripeHeight * BytesPerPixel(_pixelFormat));    

    return res;
}

template<>
std::shared_ptr<IBitmap> PpmDecoder::ReadBinaryStripe<2>(uint32_t stripeHeight)
{
    auto res = CreateStripe(stripeHeight);

    for (uint32_t i = 0; i < stripeHeight; ++i)
    {
        auto pScanline = res->GetPlanarScanline(i);
        for (uint32_t j = 0; j < _width * ChannelCount(_pixelFormat); ++j)
        {
            char bytes[2];
            _pStream->read(bytes, 2);
            *pScanline++ = bytes[1];
            *pScanline++ = bytes[0];
        }
    }

    return res;
}

std::shared_ptr<IBitmap> PpmDecoder::ReadTextStripe(uint32_t stripeHeight)
{
    auto pStripe = CreateStripe(stripeHeight);

    for (uint32_t i = 0; i < stripeHeight; ++i)
    {
        for (uint32_t j = 0; j < _width; ++j)
        {
            for (uint32_t k = 0; k < ChannelCount(_pixelFormat); ++k)
            {
                uint32_t v;
                *_pStream >> v;
                pStripe->SetChannel(i, j, k, v);
            }
        }
    }

    return pStripe;
}


std::shared_ptr<IBitmap> PpmDecoder::ReadBitmap()
{
    if (!_pStream)
        throw std::runtime_error("decoder is detached");

    _pStream->seekg(_dataOffset, std::ios_base::beg);
    _currentScanline = 0;
    return ReadStripe();
}

std::shared_ptr<IBitmap> PpmDecoder::ReadStripe(uint32_t stripeHeight)
{
    if (!_pStream)
        throw std::runtime_error("decoder is detached");

    if (stripeHeight > _height - _currentScanline)
        throw std::invalid_argument("stripe height exceeds the remainder");

    if (stripeHeight == 0)
        stripeHeight = _height - _currentScanline;

    std::shared_ptr<IBitmap> pRes;
    if (_ppmMode == PpmMode::Text)
        pRes = ReadTextStripe(stripeHeight);
    else 
        pRes = BytesPerChannel(_pixelFormat) == 1 ? ReadBinaryStripe<1>(stripeHeight) : ReadBinaryStripe<2>(stripeHeight);

    _currentScanline += stripeHeight;
    return pRes;
}

uint32_t PpmDecoder::GetCurrentScanline() const
{
    return _currentScanline;
}

std::shared_ptr<IBitmap> PpmDecoder::CreateStripe(uint32_t stripeHeight)
{
    switch(_pixelFormat)
    {
    case PixelFormat::Gray8:
        return std::make_shared<Bitmap<PixelFormat::Gray8>>(_width, stripeHeight);
    case PixelFormat::Gray16:
        return std::make_shared<Bitmap<PixelFormat::Gray16>>(_width, stripeHeight);
    case PixelFormat::RGB24:
        return std::make_shared<Bitmap<PixelFormat::RGB24>>(_width, stripeHeight);
    case PixelFormat::RGB48:
        return std::make_shared<Bitmap<PixelFormat::RGB48>>(_width, stripeHeight);
    default:
        throw std::runtime_error("not implemented");
    }
}



std::unique_ptr<std::istringstream> PpmDecoder::ReadLine()
{
    auto res = ImageDecoder::ReadLine();
    while (res->peek() == '#')
        res = ImageDecoder::ReadLine();

    return res;
}
