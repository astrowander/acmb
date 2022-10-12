#include "test.h"
#include "testtools.h"
#include "../Core/camerasettings.h"
#include "../Transforms/DebayerTransform.h"

ACMB_TESTS_NAMESPACE_BEGIN

BEGIN_SUITE( DebayerTransform )

BEGIN_TEST( TestNullArgs )

auto f1 = []
{
    DebayerTransform::Create( nullptr, std::make_shared<CameraSettings>() );
};

auto f2 = []
{
    DebayerTransform::Create( IBitmap::Create( 10, 10, PixelFormat::Gray16 ), nullptr );
};

auto f3 = []
{
    DebayerTransform::Create( PixelFormat::Gray16, nullptr );
};

ASSERT_THROWS( f1, std::invalid_argument );
ASSERT_THROWS( f2, std::invalid_argument );
ASSERT_THROWS( f3, std::invalid_argument );
END_TEST

BEGIN_TEST( TestGray16 )

auto pBitmap = IBitmap::Create( GetPathToTestFile( "PPM/bayer.pgm" ) );
auto pCameraSettings = std::make_shared<CameraSettings>();
pCameraSettings->maxChannel = 15340;
pCameraSettings->channelPremultipiers = { 2.09313083f, 0.943063855f, 1.34144759f, 0.0f };
auto pDebayer = DebayerTransform::Create( pBitmap, pCameraSettings );
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "DebayerTransform/TestGray16.ppm" ), pDebayer->RunAndGetBitmap() ) );
END_TEST

BEGIN_TEST( TestGray8 )
auto f = []
{
    auto pBitmap = IBitmap::Create( 10, 10, PixelFormat::Gray8);   
    auto pDebayer = DebayerTransform::Create( pBitmap, std::make_shared<CameraSettings>() );
};

ASSERT_THROWS( f, std::invalid_argument );
END_TEST

BEGIN_TEST( TestRGB24 )
auto f = []
{
    auto pBitmap = IBitmap::Create( 10, 10, PixelFormat::RGB24 );
    auto pDebayer = DebayerTransform::Create( pBitmap, std::make_shared<CameraSettings>() );
};

ASSERT_THROWS( f, std::invalid_argument );
END_TEST

BEGIN_TEST( TestRGB48 )
auto f = []
{
    auto pBitmap = IBitmap::Create( 10, 10, PixelFormat::RGB48 );
    auto pDebayer = DebayerTransform::Create( pBitmap, std::make_shared<CameraSettings>() );
};

ASSERT_THROWS( f, std::invalid_argument );
END_TEST

END_SUITE
ACMB_TESTS_NAMESPACE_END
