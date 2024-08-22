#include "test.h"
#include "testtools.h"
#include "../Transforms/CenterObjectTransform.h"
#include "../Transforms/converter.h"

ACMB_TESTS_NAMESPACE_BEGIN

BEGIN_SUITE( CenterObjectTransform )

BEGIN_TEST( TestRGB48 )
auto pSrcBitmap = IBitmap::Create( GetPathToTestFile( "PPM/jupiter.ppm" ) );
auto pDstBitmap = CenterObjectTransform::CenterObject( pSrcBitmap, { .dstSize = {.width = 300, .height = 300}, .threshold = 25 } );
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "CenterObjectTransform/TestRGB48.ppm" ), pDstBitmap ) );
END_TEST

BEGIN_TEST( TestGray8 )
auto pSrcBitmap = Converter::Convert( IBitmap::Create( GetPathToTestFile( "PPM/jupiter.ppm" ) ), PixelFormat::Gray8 );
auto pDstBitmap = CenterObjectTransform::CenterObject( pSrcBitmap, { .dstSize = {.width = 300, .height = 300}, .threshold = 25 } );
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "CenterObjectTransform/TestGray8.ppm" ), pDstBitmap ) );
END_TEST

BEGIN_TEST( TestGray16 )
auto pSrcBitmap = Converter::Convert( IBitmap::Create( GetPathToTestFile( "PPM/jupiter.ppm" ) ), PixelFormat::Gray16 );
auto pDstBitmap = CenterObjectTransform::CenterObject( pSrcBitmap, { .dstSize = {.width = 300, .height = 300}, .threshold = 25 } );
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "CenterObjectTransform/TestGray16.ppm" ), pDstBitmap ) );
END_TEST

BEGIN_TEST( TestRGB24 )
auto pSrcBitmap = Converter::Convert( IBitmap::Create( GetPathToTestFile( "PPM/jupiter.ppm" ) ), PixelFormat::RGB24 );
auto pDstBitmap = CenterObjectTransform::CenterObject( pSrcBitmap, { .dstSize = {.width = 300, .height = 300}, .threshold = 25 } );
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "CenterObjectTransform/TestRGB24.ppm" ), pDstBitmap ) );
END_TEST

BEGIN_TEST( TestUnsupportedPixelFormat )
auto f = [] ()
{
    auto pSrcBitmap = IBitmap::Create( GetPathToTestFile( "AllFormats/IMG_8944.CR2" ) ); 
    CenterObjectTransform::CenterObject( pSrcBitmap, { .dstSize = {.width = 300, .height = 300}, .threshold = 25 } );
};
ASSERT_THROWS( f, std::runtime_error );
END_TEST

BEGIN_TEST( TestNegativeThreshold )
auto f = [] ()
{
    auto pSrcBitmap = IBitmap::Create( GetPathToTestFile( "PPM/jupiter.ppm" ) ); 
    CenterObjectTransform::CenterObject( pSrcBitmap, { .dstSize = {.width = 300, .height = 300}, .threshold = -1 } );
};
ASSERT_THROWS( f, std::invalid_argument );
END_TEST

BEGIN_TEST( TestTooLargeSize )
    auto pSrcBitmap = IBitmap::Create( GetPathToTestFile( "PPM/jupiter_crop.ppm" ) ); 
    EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "CenterObjectTransform/TestTooLargeSize.ppm" ), CenterObjectTransform::CenterObject( pSrcBitmap, { .dstSize = {.width = 600, .height = 400}, .threshold = 25 } ) ) );
END_TEST

BEGIN_TEST( TestZeroSize )
    auto f = [] ()
    {
        auto pSrcBitmap = IBitmap::Create( GetPathToTestFile( "PPM/jupiter.ppm" ) );
        CenterObjectTransform::CenterObject( pSrcBitmap, { .dstSize = {.width = 0, .height = 0}, .threshold = 25 } );
    };
    ASSERT_THROWS( f, std::invalid_argument );
END_TEST

BEGIN_TEST( TestNullArg )
    auto f = [] ()
    {
        CenterObjectTransform::CenterObject( nullptr, { .dstSize = {.width = 300, .height = 300}, .threshold = 25 } );
    };
    ASSERT_THROWS( f, std::invalid_argument );
END_TEST

END_SUITE

ACMB_TESTS_NAMESPACE_END