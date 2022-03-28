#include "RawDecoder.h"
#include "libraw/libraw.h"
#include "../../Core/bitmap.h"
#include <map>
#include <string>

RawDecoder::RawDecoder(bool halfSize)
: _pLibRaw(new LibRaw())
, _halfSize(halfSize)
{

}

const double cropFactors[9] =
{
	0.0,
	1.6,
	1.0,
	28.7,
	0.8,
	2.73,
	0,
	0,
	2.08
};

const SizeF sensorSizes[9] =
{
	{},
	{22.3, 14.9}, //APS-C
	{36.0, 24.0}, //FF
	{43.8, 23.9}, //MF
	{28.7, 19.0}, //APS-H
	{13.2, 8.8}, //1 inch
	{},
	{},
	{17.3, 13.0} // 4/3
};

const std::map<uint32_t, std::string> lensMakers =
{
	{180, "Sigma"}
};

const std::map<uint32_t, std::string> lensNames =
{
	{180, "Sigma 24mm f/1.4 DG HSM [A]"}
};

void RawDecoder::Attach(const std::string& fileName)
{
	_lastFileName = fileName;

	if (_pLibRaw->open_file(fileName.data()))
		throw std::runtime_error("unable to read the file");

    _pLibRaw->imgdata.params.output_bps = 16;
    _pLibRaw->imgdata.params.no_interpolation = 0;
    _pLibRaw->imgdata.params.fbdd_noiserd = 0;
    _pLibRaw->imgdata.params.med_passes = 0;
    _pLibRaw->imgdata.params.half_size = _halfSize;

	_pLibRaw->raw2image_start();
	_width = _pLibRaw->imgdata.sizes.flip < 5 ? _pLibRaw->imgdata.sizes.iwidth : _pLibRaw->imgdata.sizes.iheight;
	_height = _pLibRaw->imgdata.sizes.flip < 5 ? _pLibRaw->imgdata.sizes.iheight : _pLibRaw->imgdata.sizes.iwidth;

	_pCameraSettings->timestamp = _pLibRaw->imgdata.other.timestamp;
	_pCameraSettings->sensorSizeMm = sensorSizes[_pLibRaw->imgdata.lens.makernotes.CameraFormat];
	_pCameraSettings->cropFactor = cropFactors[_pLibRaw->imgdata.lens.makernotes.CameraFormat];
	_pCameraSettings->focalLength = _pLibRaw->imgdata.other.focal_len;
	_pCameraSettings->radiansPerPixel = 2 * atan(_pCameraSettings->sensorSizeMm.height / (2 * _pCameraSettings->focalLength)) / _height;	
	_pCameraSettings->aperture = _pLibRaw->imgdata.other.aperture;

	_pCameraSettings->cameraMakerName = _pLibRaw->imgdata.idata.make;
	_pCameraSettings->cameraModelName = _pCameraSettings->cameraMakerName;
	_pCameraSettings->cameraModelName.push_back(' ');
	_pCameraSettings->cameraModelName.append(_pLibRaw->imgdata.idata.model);

	if (lensMakers.find(_pLibRaw->imgdata.lens.makernotes.LensID) != std::end(lensMakers))
		_pCameraSettings->lensMakerName = lensMakers.at(_pLibRaw->imgdata.lens.makernotes.LensID);

	if (lensNames.find(_pLibRaw->imgdata.lens.makernotes.LensID) != std::end(lensNames))
		_pCameraSettings->lensModelName = lensNames.at(_pLibRaw->imgdata.lens.makernotes.LensID);

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

