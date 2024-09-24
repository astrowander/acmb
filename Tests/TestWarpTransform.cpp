#include "test.h"
#include "testtools.h"
#include "../Transforms/WarpTransform.h"

ACMB_TESTS_NAMESPACE_BEGIN

BEGIN_SUITE( WarpTransform )

BEGIN_TEST( TestRGB24 )
auto pSrcBitmap = IBitmap::Create( GetPathToTestFile( "PPM/rgb24.ppm" ) );
WarpTransform::Settings settings;
settings.controls[1].y = 0.2;
settings.controls[2].y = 0.2;

auto pDstBitmap = WarpTransform::Warp( pSrcBitmap, settings );
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "WarpTransform/TestRGB24.ppm" ), pDstBitmap ) );
END_TEST

END_SUITE

ACMB_TESTS_NAMESPACE_END