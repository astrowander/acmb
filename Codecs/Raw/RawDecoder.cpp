#include "RawDecoder.h"
#include "libraw/libraw.h"
#include "../../Core/bitmap.h"

RawDecoder::RawDecoder()
: _pLibRaw(new LibRaw())
{

}

void RawDecoder::Attach(const std::string& fileName)
{
	_pLibRaw->open_file(fileName.data());
	_pLibRaw->raw2image_start();
	_width = _pLibRaw->imgdata.sizes.width;
	_height = _pLibRaw->imgdata.sizes.height;
	_pixelFormat = _pLibRaw->imgdata.params.output_bps == 8 ? PixelFormat::RGB24 : PixelFormat::RGB48;
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
	ret = _pLibRaw->dcraw_process();
	libraw_processed_image_t* image = _pLibRaw->dcraw_make_mem_image(&ret);
	std::memcpy(pRes->GetPlanarScanline(0), image->data, image->data_size);
	_pLibRaw->dcraw_clear_mem(image);
	_pLibRaw->free_image();
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
