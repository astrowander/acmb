#include "test.h"
#include "testtools.h"
#include "../Transforms/DebayerTransform.h"

ACMB_TESTS_NAMESPACE_BEGIN

BEGIN_SUITE( DebayerTransform )

BEGIN_TEST( TestGray16 )

auto pBitmap = IBitmap::Create( GetPathToTestFile( "PPM/bayer.pgm" ) );
auto pDebayer = DebayerTransform::Create( pBitmap, nullptr );
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "DebayerTransform/TestGray16.ppm" ), pDebayer->RunAndGetBitmap() ) );
END_TEST

END_SUITE
ACMB_TESTS_NAMESPACE_END