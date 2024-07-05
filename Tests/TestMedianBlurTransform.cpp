#include "test.h"
#include "testtools.h"
#include "../Transforms/MedianBlurTransform.h"
#include "../Transforms/converter.h"

ACMB_TESTS_NAMESPACE_BEGIN

BEGIN_SUITE( MedianBlurTransform )

BEGIN_TEST( TestRGB24 )
auto pSrcBitmap = IBitmap::Create( GetPathToTestFile( "PPM/rgb24.ppm" ) );
auto pDstBitmap = MedianBlurTransform::MedianBlur( pSrcBitmap, 3 );
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "MedianBlurTransform/TestRGB24.ppm" ), pDstBitmap ) );
END_TEST

BEGIN_TEST( TestGray8 )
auto pSrcBitmap = Converter::Convert( IBitmap::Create( GetPathToTestFile( "PPM/rgb24.ppm" ) ), PixelFormat::Gray8 );
auto pDstBitmap = MedianBlurTransform::MedianBlur( pSrcBitmap, 3 );
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "MedianBlurTransform/TestGray8.ppm" ), pDstBitmap ) );
END_TEST

BEGIN_TEST( TestGray16 )
auto pSrcBitmap = Converter::Convert( IBitmap::Create( GetPathToTestFile( "PPM/rgb24.ppm" ) ), PixelFormat::Gray16 );
auto pDstBitmap = MedianBlurTransform::MedianBlur( pSrcBitmap, 3 );
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "MedianBlurTransform/TestGray16.ppm" ), pDstBitmap ) );
END_TEST

BEGIN_TEST( TestRGB48 )
auto pSrcBitmap = Converter::Convert( IBitmap::Create( GetPathToTestFile( "PPM/rgb24.ppm" ) ), PixelFormat::RGB48 );
auto pDstBitmap = MedianBlurTransform::MedianBlur( pSrcBitmap, 3 );
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "MedianBlurTransform/TestRGB48.ppm" ), pDstBitmap ) );
END_TEST

BEGIN_TEST( TestUnsupportedPixelFormat )
auto f = [] ()
{
    auto pSrcBitmap = IBitmap::Create( GetPathToTestFile( "SER/19_45_16.ser" ) );
    MedianBlurTransform::MedianBlur( pSrcBitmap, 3 );
};
ASSERT_THROWS( f, std::invalid_argument );
END_TEST

BEGIN_TEST( TestNegativeRadius )
auto f = [] ()
{
    auto pSrcBitmap = IBitmap::Create( GetPathToTestFile( "PPM/rgb24.ppm" ) );
    MedianBlurTransform::MedianBlur( pSrcBitmap, -1 );
};
ASSERT_THROWS( f, std::invalid_argument );
END_TEST

BEGIN_TEST( TestZeroRadius )
auto f = [] ()
{
    auto pSrcBitmap = IBitmap::Create( GetPathToTestFile( "PPM/rgb24.ppm" ) );
    MedianBlurTransform::MedianBlur( pSrcBitmap, 0 );
};
ASSERT_THROWS( f, std::invalid_argument );
END_TEST

BEGIN_TEST( TestUnitRadius )
    auto pSrcBitmap = IBitmap::Create( GetPathToTestFile( "PPM/rgb24.ppm" ) );
    auto pDstBitmap = MedianBlurTransform::MedianBlur( pSrcBitmap, 1 );
    EXPECT_TRUE( BitmapsAreEqual( pSrcBitmap, pDstBitmap ) );
END_TEST

BEGIN_TEST( TestLargeRadius )
    auto pSrcBitmap = IBitmap::Create( GetPathToTestFile( "PPM/rgb24.ppm" ) );
    auto pDstBitmap = MedianBlurTransform::MedianBlur( pSrcBitmap, 100 );
    EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "MedianBlurTransform/TestLargeRadius.ppm" ), pDstBitmap ) );
END_TEST

BEGIN_TEST( TestNullBitmap )
auto f = [] ()
{
    MedianBlurTransform::MedianBlur( nullptr, 3 );
};
ASSERT_THROWS( f, std::invalid_argument );
END_TEST

END_SUITE

ACMB_TESTS_NAMESPACE_END