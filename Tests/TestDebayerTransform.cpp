#include "test.h"
#include "testtools.h"
#include "../Core/CameraSettings.h"
#include "../Transforms/DebayerTransform.h"

ACMB_TESTS_NAMESPACE_BEGIN

BEGIN_SUITE( DebayerTransform )

BEGIN_TEST( TestGray16 )

auto pBitmap = IBitmap::Create( GetPathToTestFile( "PPM/bayer.pgm" ) );
auto pCameraSettings = std::make_shared<CameraSettings>();
pCameraSettings->maxChannel = 15340;
pCameraSettings->channelPremultipiers = { 2.09313083f, 0.943063855f, 1.34144759f, 0.0f };
auto pDebayer = DebayerTransform::Create( pBitmap, pCameraSettings );
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "DebayerTransform/TestGray16.ppm" ), pDebayer->RunAndGetBitmap() ) );
END_TEST

END_SUITE
ACMB_TESTS_NAMESPACE_END