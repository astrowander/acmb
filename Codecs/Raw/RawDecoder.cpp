#include "RawDecoder.h"
#include "libraw/libraw.h"
#include "../../Core/bitmap.h"

RawDecoder::RawDecoder(bool halfSize)
: _pLibRaw(new LibRaw())
, _halfSize(halfSize)
{

}

void RawDecoder::Attach(const std::string& fileName)
{
	_pLibRaw->open_file(fileName.data());
    _pLibRaw->imgdata.params.output_bps = 16;
    _pLibRaw->imgdata.params.no_interpolation = 1;
    _pLibRaw->imgdata.params.fbdd_noiserd = 0;
    _pLibRaw->imgdata.params.med_passes = 0;
    _pLibRaw->imgdata.params.half_size = _halfSize;

	_pLibRaw->raw2image_start();
	_width = _pLibRaw->imgdata.sizes.flip < 5 ? _pLibRaw->imgdata.sizes.iwidth : _pLibRaw->imgdata.sizes.iheight;
	_height = _pLibRaw->imgdata.sizes.flip < 5 ? _pLibRaw->imgdata.sizes.iheight : _pLibRaw->imgdata.sizes.iwidth;

	_pixelFormat = PixelFormat::RGB48;
}

void RawDecoder::Attach(std::shared_ptr<std::istream> pStream)
{
	throw std::runtime_error("not implemented");		
}

std::shared_ptr<IBitmap> RawDecoder::ReadBitmap()
{
	if (!_pLibRaw)
		throw std::runtime_error("RawDecoder is detached");

	auto pRes = IBitmap::Create(_width, _height, _pixelFormat);

	auto ret = _pLibRaw->unpack();
	if (ret != LIBRAW_SUCCESS)
	{
		throw std::runtime_error("raw processing error");
	}

	ret = _pLibRaw->dcraw_process();
	if (ret != LIBRAW_SUCCESS)
	{
		_pLibRaw->recycle();
		throw std::runtime_error("raw processing error");
	}

	libraw_processed_image_t* image = _pLibRaw->dcraw_make_mem_image(&ret);
	if (ret != LIBRAW_SUCCESS)
	{
		_pLibRaw->dcraw_clear_mem(image);
		_pLibRaw->recycle();
		throw std::runtime_error("raw processing error");
	}
	std::memcpy(pRes->GetPlanarScanline(0), image->data, image->data_size);

	_pLibRaw->dcraw_clear_mem(image);
	_pLibRaw->recycle();
	return pRes;
}

std::shared_ptr<IBitmap> RawDecoder::ReadStripe(uint32_t index)
{
	throw std::runtime_error("not implemented");
}

uint32_t RawDecoder::GetCurrentScanline() const
{
	throw std::runtime_error("not implemented");
}
