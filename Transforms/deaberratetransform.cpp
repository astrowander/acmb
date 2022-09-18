#include "deaberratetransform.h"
#include "./../Core/camerasettings.h"

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
		std::vector<float> pos( lwidth );

		for ( uint32_t y = 0; y < _pSrcBitmap->GetHeight(); y++ )
		{
			if ( !_pModifier->ApplySubpixelGeometryDistortion( 0.0, y, _pSrcBitmap->GetWidth(), 1, &pos[0] ) )
				throw std::runtime_error( "unable to correct vignetting" );

			auto pDstPixel = pDstBitmap->GetScanline( y );

			for ( uint32_t x = 0; x < _pSrcBitmap->GetWidth(); ++x )
			{
				pDstPixel[0] = _pSrcBitmap->GetInterpolatedChannel( pos[6 * x], pos[6 * x + 1], 0 );
				pDstPixel[1] = _pSrcBitmap->GetInterpolatedChannel( pos[6 * x + 2], pos[6 * x + 3], 1 );
				pDstPixel[2] = _pSrcBitmap->GetInterpolatedChannel( pos[6 * x + 4], pos[6 * x + 5], 2 );
				pDstPixel += 3;
			}
		}
	}

public:
	DeaberrateTransform_( IBitmapPtr pSrcBitmap, std::shared_ptr<CameraSettings> pCameraSettings )
	: DeaberrateTransform( pSrcBitmap, pCameraSettings )
    , _pDatabase( new lfDatabase() )
	{
		if ( _pDatabase->Load() )
			throw std::runtime_error( "unable to load the lens database" );
	}

	virtual void Run() override
	{
		auto cameras = std::unique_ptr<const lfCamera*, std::function<void( void* )>>( _pDatabase->FindCameras( _pCameraSettings->cameraMakerName.c_str(), _pCameraSettings->cameraModelName.c_str() ), lf_free );
		if ( !cameras )
			throw std::runtime_error( "unable to find the camera" );

		auto pCamera = cameras.get()[0];

		auto lenses = std::unique_ptr<const lfLens*, std::function<void( void* )>>( _pDatabase->FindLenses( pCamera, _pCameraSettings->lensMakerName.c_str(), _pCameraSettings->lensModelName.c_str() ), lf_free );
		if ( !lenses )
			throw std::runtime_error( "unable to find the lens" );

		auto pLens = lenses.get()[0];

		uint32_t width = _pSrcBitmap->GetWidth();
		uint32_t height = _pSrcBitmap->GetHeight();

		_pModifier = std::make_unique<lfModifier>( pLens, _pCameraSettings->focalLength, _pCameraSettings->cropFactor, width, height, BytesPerChannel( pixelFormat ) == 1 ? LF_PF_U8 : LF_PF_U16 );

		//auto modFlags = _pModifier->Initialize(pLens, BytesPerChannel(pixelFormat) == 1 ? LF_PF_U8 : LF_PF_U16, _pCameraSettings->focalLength, _pCameraSettings->aperture, _pCameraSettings->distance, 1.0, pLens->Type, LF_MODIFY_DISTORTION | LF_MODIFY_VIGNETTING, false);

		_pDstBitmap = IBitmap::Create( width, height, pixelFormat );

		//correct vignetting if available
		if ( _pModifier->EnableVignettingCorrection( _pCameraSettings->aperture, _pCameraSettings->distance ) & LF_MODIFY_VIGNETTING )
		{
			for ( uint32_t y = 0; y < _pSrcBitmap->GetHeight(); y++ )
			{
				auto pScanline = _pSrcBitmap->GetPlanarScanline( y );
				if ( !_pModifier->ApplyColorModification( pScanline, 0.0, y, width, height, LF_CR_4( RED, GREEN, BLUE, UNKNOWN ), width * BytesPerPixel( pixelFormat ) ) )
					throw std::runtime_error( "unable to correct vignetting" );
			}
		}

		_pModifier->EnableDistortionCorrection();
		_pModifier->EnableTCACorrection();

		if ( ( _pModifier->GetModFlags() & LF_MODIFY_DISTORTION ) || ( _pModifier->GetModFlags() & LF_MODIFY_TCA ) )
		{
			CorrectDistortion();
		}
	}
};

DeaberrateTransform::DeaberrateTransform(IBitmapPtr pSrcBitmap, std::shared_ptr<CameraSettings> pCameraSettings)
: BaseTransform(pSrcBitmap)
, _pCameraSettings(pCameraSettings)

{
	
}

std::shared_ptr<DeaberrateTransform> DeaberrateTransform::Create( IBitmapPtr pSrcBitmap, std::shared_ptr<CameraSettings> pCameraSettings )
{
	switch ( pSrcBitmap->GetPixelFormat() )
	{
		case PixelFormat::RGB24:
			return std::make_shared<DeaberrateTransform_<PixelFormat::RGB24>>( pSrcBitmap, pCameraSettings );
		case PixelFormat::RGB48:
			return std::make_shared<DeaberrateTransform_<PixelFormat::RGB24>>( pSrcBitmap, pCameraSettings );
		default:
			throw std::runtime_error( "Unsupported pixel format" );
	}
}

ACMB_NAMESPACE_END
