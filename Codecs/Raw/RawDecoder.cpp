#include "RawDecoder.h"

#include "../../Core/bitmap.h"
#include "../../Tools/SystemTools.h"
#include "../JPEG/JpegDecoder.h"
#include "../../Transforms/ResizeTransform.h"

#include <string>
#include <sstream>
#include "libraw/libraw.h"
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <fstream>

#undef min

ACMB_NAMESPACE_BEGIN

RawDecoder::RawDecoder( PixelFormat outputFormat )
: ImageDecoder(outputFormat)
, _pLibRaw(new LibRaw())
{	
}

RawDecoder::LensDB RawDecoder::LoadLensDB()
{
	LensDB res;
	std::string acmbPath;
	
	try
	{
		acmbPath = GetEnv("ACMB_PATH");
	}
	catch (std::runtime_error& e)
	{
		std::cerr << e.what() << std::endl;
		return res;
	}

	std::ifstream in(acmbPath + "/Codecs/Raw/lensdb.txt");
	if (!in.is_open())
		return res;

	while ( !in.eof() )
	{
		std::string line;
		std::getline( in, line );

		const auto pointPos = line.find_first_of( '.' );
		const auto tabPos = line.find_first_of( '\t' );
		const auto spacePos = line.find_first_of( ' ' );
		const auto key = uint16_t( std::stoi( line.substr( 0, std::min( pointPos, tabPos ) ) ) );

		res[key].emplace_back();
		auto& info = res[key].back();

		const auto endPos = line.find( " or" );
		info.fullName = line.substr( spacePos + 1, endPos - spacePos - 1 );

		const auto focalStart = info.fullName.find_first_of( "1234567890" );
		const auto focalEnd = info.fullName.find_first_of( 'm', focalStart );
		const auto minusPos = info.fullName.find_first_of( '-', focalStart );
		if ( focalStart == std::string::npos || focalEnd == std::string::npos )
			continue;

		if ( minusPos < focalEnd )
		{
			info.minFocal = std::stoi( info.fullName.substr( focalStart, minusPos - focalStart ) );
			info.maxFocal = std::stoi( info.fullName.substr( minusPos + 1, focalEnd - minusPos - 1 ) );
		}
		else
		{
			info.minFocal = info.maxFocal = std::stoi( info.fullName.substr( focalStart, focalEnd - focalStart ) );
		}
	}

	return res;
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

void RawDecoder::Attach()
{
	_decodedFormat = PixelFormat::Bayer16;

	_pLibRaw->imgdata.params.output_bps = BitsPerChannel( _decodedFormat );
	_pLibRaw->imgdata.params.no_interpolation = 1;
	_pLibRaw->imgdata.params.fbdd_noiserd = 0;
	_pLibRaw->imgdata.params.med_passes = 0;
	_pLibRaw->imgdata.params.no_auto_bright = 1;
	_pLibRaw->imgdata.params.half_size = 0;
    _pLibRaw->imgdata.params.user_qual = 0;
    _pLibRaw->imgdata.rawparams.use_rawspeed = 1;

	_pLibRaw->raw2image_start();
	_width = _pLibRaw->imgdata.sizes.iwidth;
	_height = _pLibRaw->imgdata.sizes.iheight;

	if ( !_pCameraSettings )
		_pCameraSettings = std::make_shared<CameraSettings>();

	_pCameraSettings->timestamp = _pLibRaw->imgdata.other.timestamp;
	_pCameraSettings->sensorSizeMm = sensorSizes[_pLibRaw->imgdata.lens.makernotes.CameraFormat];
	_pCameraSettings->cropFactor = cropFactors[_pLibRaw->imgdata.lens.makernotes.CameraFormat];
	_pCameraSettings->focalLength = _pLibRaw->imgdata.other.focal_len;
	_pCameraSettings->radiansPerPixel = 2 * atan( _pCameraSettings->sensorSizeMm.height / ( 2 * _pCameraSettings->focalLength ) ) / _height;
	_pCameraSettings->aperture = _pLibRaw->imgdata.other.aperture;

	_pCameraSettings->cameraMakerName = _pLibRaw->imgdata.idata.make;
	_pCameraSettings->cameraModelName = _pCameraSettings->cameraMakerName;
	_pCameraSettings->cameraModelName.push_back( ' ' );
	_pCameraSettings->cameraModelName.append( _pLibRaw->imgdata.idata.model );

	_pCameraSettings->maxChannel = _pLibRaw->imgdata.color.maximum;
	for ( int i = 0; i < 4; ++i )
	{
		_pCameraSettings->channelPremultipiers[i] = _pLibRaw->imgdata.color.pre_mul[i];
	}

	if ( lensDB.contains( _pLibRaw->imgdata.lens.makernotes.LensID ) )
	{

		for ( const auto& candidate : lensDB.at( _pLibRaw->imgdata.lens.makernotes.LensID ) )
		{
			if ( int( _pCameraSettings->focalLength ) >= candidate.minFocal && int( _pCameraSettings->focalLength ) <= candidate.maxFocal )
			{
				_pCameraSettings->lensMakerName = candidate.fullName.substr( 0, candidate.fullName.find_first_of( ' ' ) );
				_pCameraSettings->lensModelName = candidate.fullName;
				break;
			}
		}
	}

	if ( _pixelFormat == PixelFormat::Unspecified )
		_pixelFormat = _decodedFormat;
}

void RawDecoder::Attach(const std::string& fileName)
{
	_lastFileName = fileName;

	if (_pLibRaw->open_file(fileName.data()))
		throw std::runtime_error("unable to read the file");

	Attach();
}

void RawDecoder::Attach(std::shared_ptr<std::istream> is)
{
    //std::vector<char> buf;
	is->seekg( 0, is->end );
	size_t length = is->tellg();
	is->seekg( 0, is->beg );

    _buf.resize( length );
    is->read( _buf.data(), length );

	if ( !is )
		throw std::runtime_error( "unable to read the stream" );

    if ( _pLibRaw->open_buffer( _buf.data(), _buf.size() ) )
		throw std::runtime_error( "unable to read the file" );

	Attach();	
}

void RawDecoder::Detach()
{
    _pLibRaw->recycle();
    _buf.clear();
}

std::shared_ptr<IBitmap> RawDecoder::ReadBitmap()
{
	if (!_pLibRaw)
		throw std::runtime_error("RawDecoder is detached");

	auto ret = _pLibRaw->unpack();
	if (ret != LIBRAW_SUCCESS)
	{
		throw std::runtime_error("raw processing error");
	}

	if ( _pLibRaw->imgdata.rawdata.raw_image )
	{
        _decodedFormat = PixelFormat::Bayer16;
        if ( _pixelFormat == PixelFormat::Unspecified )
            _pixelFormat = _decodedFormat;
	}
	else if ( _pLibRaw->imgdata.rawdata.color4_image )
	{
		_decodedFormat = PixelFormat::RGBA64;
		if ( _pixelFormat == PixelFormat::Bayer16 )
            _pixelFormat = PixelFormat::RGB48;
	}
        
	auto pRes = IBitmap::Create( _width, _height, _decodedFormat );
	_pCameraSettings->blackLevel = _pLibRaw->imgdata.color.black;
	
	const int rawStride = _pLibRaw->imgdata.sizes.raw_width * BytesPerPixel( _decodedFormat );
	const int topOffset = rawStride * _pLibRaw->imgdata.sizes.top_margin;
	const int leftOffset = _pLibRaw->imgdata.sizes.left_margin * BytesPerPixel( _decodedFormat );
	const int stride = _width * BytesPerPixel( _decodedFormat );

	if ( _pLibRaw->imgdata.rawdata.raw_image )
	{
		oneapi::tbb::parallel_for( oneapi::tbb::blocked_range<int>( 0, _height ), [&] ( const oneapi::tbb::blocked_range<int>& range )
		{
			for ( int i = range.begin(); i < range.end(); ++i )
			{
				memcpy( pRes->GetPlanarScanline( i ), reinterpret_cast< char* >(_pLibRaw->imgdata.rawdata.raw_image) + topOffset + i * rawStride + leftOffset, stride );
			}
		} );
	}
	else if ( _pLibRaw->imgdata.rawdata.color4_image )
	{
		oneapi::tbb::parallel_for( oneapi::tbb::blocked_range<int>( 0, _height ), [&] ( const oneapi::tbb::blocked_range<int>& range )
		{
			for ( int i = range.begin(); i < range.end(); ++i )
			{
				memcpy( pRes->GetPlanarScanline( i ), reinterpret_cast< char* >(_pLibRaw->imgdata.rawdata.color4_image) + topOffset + i * rawStride + leftOffset, stride );
			}
		} );
	}
	
	pRes->SetCameraSettings( _pCameraSettings );
    pRes = ToOutputFormat( pRes );
	Reattach();
	return pRes;
}

std::shared_ptr<IBitmap> RawDecoder::ReadPreview()
{
    if ( !_pLibRaw )
        throw std::runtime_error( "RawDecoder is detached" );

    auto oldPixelFormat = _pixelFormat;
    _pixelFormat = PixelFormat::RGB48;
    auto pRes = ReadBitmap();
    _pixelFormat = oldPixelFormat;

	Size previewSize{ int( pRes->GetWidth() ), int( pRes->GetHeight() ) };
	if ( previewSize.width > 1280 || previewSize.height > 720 )
		pRes = ResizeTransform::Resize( pRes, ResizeTransform::GetSizeWithPreservedRatio( previewSize, { 1280, 720 } ) );
	return pRes;
	/*if ( _pLibRaw->imgdata.thumbs_list.thumbcount == 0 )
		return ImageDecoder::ReadPreview();

    if ( _pLibRaw->unpack_thumb() != LIBRAW_SUCCESS )
		throw std::runtime_error( "raw processing error" );

	IBitmapPtr pPreviewBitmap;
	if ( _pLibRaw->imgdata.thumbnail.tformat == LIBRAW_THUMBNAIL_JPEG )
	{
        auto ss = std::make_shared<std::istringstream>( std::string( _pLibRaw->imgdata.thumbnail.thumb, _pLibRaw->imgdata.thumbnail.tlength ), std::ios_base::binary );
        JpegDecoder decoder;
        decoder.Attach( ss );
		pPreviewBitmap = decoder.ReadBitmap();
		decoder.Detach();
	}
	else if ( _pLibRaw->imgdata.thumbnail.tformat == LIBRAW_THUMBNAIL_BITMAP16 )
	{
        pPreviewBitmap = IBitmap::Create( _pLibRaw->imgdata.thumbnail.twidth, _pLibRaw->imgdata.thumbnail.theight, PixelFormat::RGB48 );
        std::copy( _pLibRaw->imgdata.thumbnail.thumb, _pLibRaw->imgdata.thumbnail.thumb +_pLibRaw->imgdata.thumbnail.tlength, (char*)pPreviewBitmap->GetPlanarScanline( 0 ) );
	}
	else if (_pLibRaw->imgdata.thumbnail.tformat == LIBRAW_THUMBNAIL_BITMAP)
	{
        pPreviewBitmap = IBitmap::Create( _pLibRaw->imgdata.thumbnail.twidth, _pLibRaw->imgdata.thumbnail.theight, PixelFormat::RGB24 );
        std::copy( _pLibRaw->imgdata.thumbnail.thumb, _pLibRaw->imgdata.thumbnail.thumb + _pLibRaw->imgdata.thumbnail.tlength, ( char* ) pPreviewBitmap->GetPlanarScanline( 0 ) );
	}
	else
    {
        return ImageDecoder::ReadPreview();
    }

    Size previewSize{ int( pPreviewBitmap->GetWidth() ), int( pPreviewBitmap->GetHeight() ) };
	if ( previewSize.width > 1280 || previewSize.height > 720 )
		pPreviewBitmap = ResizeTransform::Resize( pPreviewBitmap, ResizeTransform::GetSizeWithPreservedRatio( previewSize, { 1280, 720 } ) );
	return pPreviewBitmap;*/
}

std::unordered_set<std::string> RawDecoder::GetExtensions()
{
	return { ".ari", ".dpx", ".arw", ".srf", ".sr2", ".bay", ".cr3", ".crw", ".cr2", ".dng", ".dcr", ".kdc", ".erf", ".3fr", ".mef", ".mrw", ".nef", ".nrw", ".orf", ".ptx", ".pef", ".raf", ".raw", ".rw1", ".rw2", "r3d", "srw", ".x3f" };
}

ACMB_NAMESPACE_END
