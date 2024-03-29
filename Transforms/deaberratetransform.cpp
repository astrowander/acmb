#include "deaberratetransform.h"
#include "./../Core/camerasettings.h"
#include "./../Tools/SystemTools.h"

#include "lensfun/lensfun.h"
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/enumerable_thread_specific.h>


ACMB_NAMESPACE_BEGIN

template<PixelFormat pixelFormat>
class DeaberrateTransform_ final: public DeaberrateTransform
{
	std::unique_ptr<lfDatabase> _pDatabase;
	std::unique_ptr<lfModifier> _pModifier;

	void CorrectDistortion()
	{
		auto pSrcBitmap = std::static_pointer_cast< Bitmap<pixelFormat> >( _pSrcBitmap );
		auto pDstBitmap = std::static_pointer_cast< Bitmap<pixelFormat> >( _pDstBitmap );

		int lwidth = _pSrcBitmap->GetWidth() * 2 * PixelFormatTraits<pixelFormat>::channelCount;
		tbb::enumerable_thread_specific<std::vector<float>> buffer(lwidth);
		oneapi::tbb::parallel_for( oneapi::tbb::blocked_range<int>( 0, pSrcBitmap->GetHeight() ), [&] ( const oneapi::tbb::blocked_range<int>& range )
		{
			for ( int i = range.begin(); i < range.end(); ++i )
			{				
				auto& pos = buffer.local();
				if constexpr ( GetColorSpace( pixelFormat ) == ColorSpace::RGB )
				{
					if ( !_pModifier->ApplySubpixelGeometryDistortion( 0.0, i, pSrcBitmap->GetWidth(), 1, &pos[0] ) )
						throw std::runtime_error( "unable to correct distortion" );

					auto pDstPixel = pDstBitmap->GetScanline( i );

					for ( uint32_t x = 0; x < _pSrcBitmap->GetWidth(); ++x )
					{
						pDstPixel[0] = pSrcBitmap->GetInterpolatedChannel( pos[6 * x], pos[6 * x + 1], 0 );
						pDstPixel[1] = pSrcBitmap->GetInterpolatedChannel( pos[6 * x + 2], pos[6 * x + 3], 1 );
						pDstPixel[2] = pSrcBitmap->GetInterpolatedChannel( pos[6 * x + 4], pos[6 * x + 5], 2 );
						pDstPixel += 3;
					}
				}
				else //Grayscale
				{
					if ( !_pModifier->ApplyGeometryDistortion( 0.0, i, pSrcBitmap->GetWidth(), 1, &pos[0] ) )
						throw std::runtime_error( "unable to correct distortion" );

					auto pDstPixel = pDstBitmap->GetScanline( i );

					for ( uint32_t x = 0; x < _pSrcBitmap->GetWidth(); ++x )
					{
						pDstPixel[0] = pSrcBitmap->GetInterpolatedChannel( pos[2 * x], pos[2 * x + 1], 0 );
						++pDstPixel;
					}
				}
			}
		} );
	}

public:
	DeaberrateTransform_( IBitmapPtr pSrcBitmap, std::shared_ptr<CameraSettings> pCameraSettings )
	: DeaberrateTransform( pSrcBitmap, pCameraSettings )
    , _pDatabase( new lfDatabase() )
	{
		auto path = GetEnv( "ACMB_PATH" ) + "/Libs/lensfun/data/db";
		auto res = _pDatabase->Load( path.c_str() );
		if ( res )
			throw std::runtime_error( "unable to load the lens database" );
	}

	virtual void Run() override
	{
		auto cameras = std::unique_ptr<const lfCamera*, std::function<void( void* )>>( _pDatabase->FindCameras( _pCameraSettings->cameraMakerName.c_str(), _pCameraSettings->cameraModelName.c_str() ), lf_free );
		if ( !cameras )
		{
			std::cout << "unable to find the camera, skip deaberration" << std::endl;
			_pDstBitmap = _pSrcBitmap;
			return;
		}

		auto pCamera = cameras.get()[0];

		auto lenses = std::unique_ptr<const lfLens*, std::function<void( void* )>>( _pDatabase->FindLenses( pCamera, _pCameraSettings->lensMakerName.c_str(), _pCameraSettings->lensModelName.c_str() ), lf_free );
		if ( !lenses )
		{
			std::cout << "unable to find the lens, skip deaberration" << std::endl;
			_pDstBitmap = _pSrcBitmap;
			return;
		}

		auto pLens = lenses.get()[0];

		uint32_t width = _pSrcBitmap->GetWidth();
		uint32_t height = _pSrcBitmap->GetHeight();

		_pModifier = std::make_unique<lfModifier>( pLens, _pCameraSettings->focalLength, _pCameraSettings->cropFactor, width, height, BytesPerChannel( pixelFormat ) == 1 ? LF_PF_U8 : LF_PF_U16 );

		//auto modFlags = _pModifier->Initialize(pLens, BytesPerChannel(pixelFormat) == 1 ? LF_PF_U8 : LF_PF_U16, _pCameraSettings->focalLength, _pCameraSettings->aperture, _pCameraSettings->distance, 1.0, pLens->Type, LF_MODIFY_DISTORTION | LF_MODIFY_VIGNETTING, false);

		_pDstBitmap = IBitmap::Create( width, height, pixelFormat );

		//correct vignetting if available
		if ( _pModifier->EnableVignettingCorrection( _pCameraSettings->aperture, _pCameraSettings->distance ) & LF_MODIFY_VIGNETTING )
		{
            oneapi::tbb::parallel_for( oneapi::tbb::blocked_range<int>( 0, _pSrcBitmap->GetHeight() ), [this, width] ( const oneapi::tbb::blocked_range<int>& range )
			{
				for ( int i = range.begin(); i < range.end(); ++i )
				{
					auto pScanline = _pSrcBitmap->GetPlanarScanline( i );
					if constexpr ( GetColorSpace( pixelFormat ) == ColorSpace::RGB )
					{
						if ( !_pModifier->ApplyColorModification( pScanline, 0.0, i, width, 1, LF_CR_3( RED, GREEN, BLUE ), width * BytesPerPixel( pixelFormat ) ) )
							throw std::runtime_error( "unable to correct vignetting" );
					}
					else // Gray
					{
						if ( !_pModifier->ApplyColorModification( pScanline, 0.0, i, width, 1, LF_CR_1( INTENSITY ), width * BytesPerPixel( pixelFormat ) ) )
							throw std::runtime_error( "unable to correct vignetting" );
					}
				}
			} );
		}

		_pModifier->EnableDistortionCorrection();
		_pModifier->EnableTCACorrection();

		if ( ( _pModifier->GetModFlags() & LF_MODIFY_DISTORTION ) || ( _pModifier->GetModFlags() & LF_MODIFY_TCA ) )
		{
			CorrectDistortion();
		}
	}

	virtual void ValidateSettings() override
	{		
	}
};

DeaberrateTransform::DeaberrateTransform(IBitmapPtr pSrcBitmap, std::shared_ptr<CameraSettings> pCameraSettings)
: BaseTransform(pSrcBitmap)
, _pCameraSettings(pCameraSettings)
{
	
}

std::shared_ptr<DeaberrateTransform> DeaberrateTransform::Create( IBitmapPtr pSrcBitmap, std::shared_ptr<CameraSettings> pCameraSettings )
{
	if ( !pSrcBitmap )
		throw std::invalid_argument( "pSrcBitmap is null" );

	if ( !pCameraSettings )
		throw std::invalid_argument( "pCameraSettings is null" );

	switch ( pSrcBitmap->GetPixelFormat() )
	{
		case PixelFormat::RGB24:
			return std::make_shared<DeaberrateTransform_<PixelFormat::RGB24>>( pSrcBitmap, pCameraSettings );
		case PixelFormat::RGB48:
			return std::make_shared<DeaberrateTransform_<PixelFormat::RGB48>>( pSrcBitmap, pCameraSettings );
		case PixelFormat::Gray8:
			return std::make_shared<DeaberrateTransform_<PixelFormat::Gray8>>( pSrcBitmap, pCameraSettings );
		case PixelFormat::Gray16:
			return std::make_shared<DeaberrateTransform_<PixelFormat::Gray16>>( pSrcBitmap, pCameraSettings );
		default:
			throw std::runtime_error( "Unsupported pixel format" );
	}
}

std::shared_ptr<DeaberrateTransform> DeaberrateTransform::Create( PixelFormat pixelFormat, std::shared_ptr<CameraSettings> pCameraSettings )
{
	if ( !pCameraSettings )
		throw std::invalid_argument( "pCameraSettings is null" );

	switch ( pixelFormat )
	{
		case PixelFormat::RGB24:
			return std::make_shared<DeaberrateTransform_<PixelFormat::RGB24>>( nullptr, pCameraSettings );
		case PixelFormat::RGB48:
			return std::make_shared<DeaberrateTransform_<PixelFormat::RGB48>>( nullptr, pCameraSettings );
		case PixelFormat::Gray8:
			return std::make_shared<DeaberrateTransform_<PixelFormat::Gray8>>( nullptr, pCameraSettings );
		case PixelFormat::Gray16:
			return std::make_shared<DeaberrateTransform_<PixelFormat::Gray16>>( nullptr, pCameraSettings );
		default:
			throw std::runtime_error( "Unsupported pixel format" );
	}
}

ACMB_NAMESPACE_END
