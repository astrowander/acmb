#include "RawDecoder.h"

#include "../../Core/bitmap.h"
#include <map>
#include <string>
#include "libraw/libraw.h"
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

ACMB_NAMESPACE_BEGIN

RawDecoder::RawDecoder( const RawSettings& rawSettings)
: _pLibRaw(new LibRaw())
, _rawSettings( rawSettings )
{
	if ( _rawSettings.outputFormat == PixelFormat::Gray8 )
		throw std::invalid_argument( "unsupported pixel format" );
}

RawDecoder::~RawDecoder()
{
	delete _pLibRaw;
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

	const bool doDebayering = ( GetColorSpace( _rawSettings.outputFormat ) == ColorSpace::RGB );

	_pLibRaw->imgdata.params.output_bps = BitsPerChannel(_rawSettings.outputFormat);
    _pLibRaw->imgdata.params.no_interpolation = int ( !doDebayering );
    _pLibRaw->imgdata.params.fbdd_noiserd = 0;
    _pLibRaw->imgdata.params.med_passes = 0;
	_pLibRaw->imgdata.params.no_auto_bright = 1;
    _pLibRaw->imgdata.params.half_size = _rawSettings.halfSize;
	_pLibRaw->imgdata.params.user_qual = 0;

	_pLibRaw->raw2image_start();
	_width = _pLibRaw->imgdata.sizes.iwidth;
	_height = _pLibRaw->imgdata.sizes.iheight;

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

	_pCameraSettings->maxChannel = _pLibRaw->imgdata.color.maximum;	
	for ( int i = 0; i < 4; ++i )
	{
		_pCameraSettings->channelPremultipiers[i] = _pLibRaw->imgdata.color.pre_mul[i];
	}

	if (lensMakers.find(_pLibRaw->imgdata.lens.makernotes.LensID) != std::end(lensMakers))
		_pCameraSettings->lensMakerName = lensMakers.at(_pLibRaw->imgdata.lens.makernotes.LensID);

	if (lensNames.find(_pLibRaw->imgdata.lens.makernotes.LensID) != std::end(lensNames))
		_pCameraSettings->lensModelName = lensNames.at(_pLibRaw->imgdata.lens.makernotes.LensID);

	_pixelFormat = _rawSettings.outputFormat;
}

void RawDecoder::Attach(std::shared_ptr<std::istream>)
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
	_pCameraSettings->blackLevel = _pLibRaw->imgdata.color.black;

	if ( _pixelFormat == PixelFormat::Gray16 )
	{
		const int rawStride = _pLibRaw->imgdata.sizes.raw_width * BytesPerPixel( _pixelFormat );
		const int topOffset = rawStride * _pLibRaw->imgdata.sizes.top_margin;
		const int leftOffset = _pLibRaw->imgdata.sizes.left_margin * BytesPerPixel( _pixelFormat );
		const int stride = _width * BytesPerPixel( _pixelFormat );

		oneapi::tbb::parallel_for( oneapi::tbb::blocked_range<int>( 0, _height ), [&] (const oneapi::tbb::blocked_range<int>& range)
		{
			for ( int i = range.begin(); i < range.end(); ++i )
			{
				memcpy( pRes->GetPlanarScanline( i ), reinterpret_cast< char* >( _pLibRaw->imgdata.rawdata.raw_image ) + topOffset + i * rawStride + leftOffset, stride );
			}
		} );
		pRes->SetCameraSettings( _pCameraSettings );
		Reattach();
		return pRes;
	}

	ret = _pLibRaw->dcraw_process();
	if (ret != LIBRAW_SUCCESS)
	{
		Detach();
		throw std::runtime_error("raw processing error");
	}

	_pLibRaw->imgdata.sizes.flip = 0;
	libraw_processed_image_t* image = _pLibRaw->dcraw_make_mem_image(&ret);
	if (ret != LIBRAW_SUCCESS)
	{
		_pLibRaw->dcraw_clear_mem(image);
		Detach();
		throw std::runtime_error("raw processing error");
	}
    memcpy(pRes->GetPlanarScanline(0), image->data, image->data_size);

	_pLibRaw->dcraw_clear_mem(image);
	Reattach();
	return pRes;
}

std::unordered_set<std::string> RawDecoder::GetExtensions()
{
	return { ".ari", ".dpx", ".arw", ".srf", ".sr2", ".bay", ".cr3", ".crw", ".cr2", ".dng", ".dcr", ".kdc", ".erf", ".3fr", ".mef", ".mrw", ".nef", ".nrw", ".orf", ".ptx", ".pef", ".raf", ".raw", ".rw1", ".rw2", "r3d", "srw", ".x3f" };
}

const RawSettings& RawDecoder::GetRawSettings()
{
	return _rawSettings;
}

void RawDecoder::SetRawSettings( const RawSettings& rawSettings )
{
	_rawSettings = rawSettings;
}

ACMB_NAMESPACE_END