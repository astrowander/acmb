#include "test.h"
#include "testtools.h"
#include "../Transforms/SaturationTransform.h"

ACMB_TESTS_NAMESPACE_BEGIN

BEGIN_SUITE( SaturationTransform )

BEGIN_TEST( TestRGB24 )
auto pSrcBitmap = IBitmap::Create( GetPathToTestFile( "PPM/rgb24.ppm" ) );
auto pDstBitmap = SaturationTransform::Saturate( pSrcBitmap, 0.5f );
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "SaturationTransform/TestRGB24.ppm" ), pDstBitmap ) );
END_TEST

BEGIN_TEST( TestRGB48 )
auto pSrcBitmap = IBitmap::Create( GetPathToTestFile( "PPM/rgb48.ppm" ) );
auto pDstBitmap = SaturationTransform::Saturate( pSrcBitmap, 0.5f );
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "SaturationTransform/TestRGB48.ppm" ), pDstBitmap ) );
END_TEST

BEGIN_TEST( TestUnsupportedPixelFormat )
auto f = [] ()
{
    auto pSrcBitmap = IBitmap::Create( GetPathToTestFile( "PPM/gray8.ppm" ) ); 
    SaturationTransform::Saturate( pSrcBitmap, 0.5f );
};
ASSERT_THROWS( f, std::invalid_argument );
END_TEST

BEGIN_TEST( TestNullArgs )
auto f = []()
{
    SaturationTransform::Saturate( nullptr, 0.5f );
};
ASSERT_THROWS( f, std::invalid_argument );
END_TEST

BEGIN_TEST( TestZeroSaturation )
auto pSrcBitmap = IBitmap::Create( GetPathToTestFile( "PPM/rgb24.ppm" ) );
auto pDstBitmap = SaturationTransform::Saturate( pSrcBitmap, 0.0f );
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "SaturationTransform/TestZeroSaturation.ppm" ), pDstBitmap ) );
END_TEST

BEGIN_TEST( TestIncreasingSaturation )
auto pSrcBitmap = IBitmap::Create( GetPathToTestFile( "PPM/rgb24.ppm" ) );
auto pDstBitmap = SaturationTransform::Saturate( pSrcBitmap, 2.0f );
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "SaturationTransform/TestIncreasingSaturation.ppm" ), pDstBitmap ) );
END_TEST

BEGIN_TEST( TestIdentitySaturation )
auto pSrcBitmap = IBitmap::Create( GetPathToTestFile( "PPM/rgb24.ppm" ) );
auto pDstBitmap = SaturationTransform::Saturate( pSrcBitmap->Clone(), 1.0f);
EXPECT_TRUE( BitmapsAreEqual( pSrcBitmap, pDstBitmap ) );
END_TEST

END_SUITE

ACMB_TESTS_NAMESPACE_END