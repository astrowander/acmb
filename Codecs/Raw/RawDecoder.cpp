#include "RawDecoder.h"
#include "libraw/libraw.h"
#include "../../Core/bitmap.h"

RawDecoder::RawDecoder(bool halfSize)
: _pLibRaw(new LibRaw())
, _halfSize(halfSize)
{

}

const SizeF sensorSizes[9] =
{
	{},
	{25.1, 16.7}, //APS-C
	{36.0, 24.0}, //FF
	{43.8, 23.9}, //MF
	{28.7, 19.0}, //APS-H
	{13.2, 8.8}, //1 inch
	{},
	{},
	{17.3, 13.0} // 4/3
};

void RawDecoder::Attach(const std::string& fileName)
{
	_lastFileName = fileName;

	_pLibRaw->open_file(fileName.data());
    _pLibRaw->imgdata.params.output_bps = 16;
    _pLibRaw->imgdata.params.no_interpolation = 0;
    _pLibRaw->imgdata.params.fbdd_noiserd = 0;
    _pLibRaw->imgdata.params.med_passes = 0;
    _pLibRaw->imgdata.params.half_size = _halfSize;

	_pLibRaw->raw2image_start();
	_width = _pLibRaw->imgdata.sizes.flip < 5 ? _pLibRaw->imgdata.sizes.iwidth : _pLibRaw->imgdata.sizes.iheight;
	_height = _pLibRaw->imgdata.sizes.flip < 5 ? _pLibRaw->imgdata.sizes.iheight : _pLibRaw->imgdata.sizes.iwidth;

	_timestamp = _pLibRaw->imgdata.other.timestamp;
	_sensorSizeMm = sensorSizes[_pLibRaw->imgdata.lens.makernotes.CameraFormat];
	_focalLength = _pLibRaw->imgdata.other.focal_len;
	_radiansPerPixel = 2 * atan(_sensorSizeMm.height / (2 * _focalLength)) / _height;

	_pixelFormat = PixelFormat::RGB48;
}

void RawDecoder::Attach(std::shared_ptr<std::istream> pStream)
{
	throw std::runtime_error("not implemented");		
}

void RawDecoder::Detach()
{
	_pLibRaw->recycle();
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
		Detach();
		throw std::runtime_error("raw processing error");
	}

	libraw_processed_image_t* image = _pLibRaw->dcraw_make_mem_image(&ret);
	if (ret != LIBRAW_SUCCESS)
	{
		_pLibRaw->dcraw_clear_mem(image);
		Detach();
		throw std::runtime_error("raw processing error");
	}
	std::memcpy(pRes->GetPlanarScanline(0), image->data, image->data_size);

	_pLibRaw->dcraw_clear_mem(image);
	Reattach();
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

