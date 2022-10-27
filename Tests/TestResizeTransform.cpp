#include "test.h"
#include "testtools.h"
#include "../Transforms/ResizeTransform.h"

ACMB_TESTS_NAMESPACE_BEGIN

BEGIN_SUITE( ResizeTransform )

BEGIN_TEST( TestRgb24 )

auto pSrcBitmap = IBitmap::Create( GetPathToTestFile( "PPM/IMG_8970.ppm" ) );
auto pResizeTransform = ResizeTransform::Create( pSrcBitmap, { 1920, 1080 } );
auto pResult = pResizeTransform->RunAndGetBitmap();

EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "ResizeTransform/TestRgb24.tif" ), pResult ) );

END_TEST
END_SUITE
ACMB_TESTS_NAMESPACE_END