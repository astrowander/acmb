#include "test.h"
#include "testtools.h"

#include "../Transforms/LevelsTransform.h"

ACMB_TESTS_NAMESPACE_BEGIN

BEGIN_SUITE( LevelsTransform )

BEGIN_TEST( TestRGB24 )

auto pBitmap = IBitmap::Create(GetPathToTestFile("PPM/rgb24.ppm"));
pBitmap = LevelsTransform::ApplyLevels( pBitmap, { .levels = { { 0.0f, 0.8f, 1.0f } } } );
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "LevelsTransform/TestRGB24.ppm" ), pBitmap ) );

END_TEST

BEGIN_TEST( TestLightening )

auto pBitmap = IBitmap::Create( GetPathToTestFile( "PPM/rgb24.ppm" ) );
pBitmap = LevelsTransform::ApplyLevels( pBitmap, { .levels = { { 0.0f, 1.5f, 1.0f } } } );
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "LevelsTransform/TestLightening.ppm" ), pBitmap ) );

END_TEST

END_SUITE

ACMB_TESTS_NAMESPACE_END