#include "test.h"
#include "testtools.h"

#include "../Transforms/ChannelEqualizer.h"

ACMB_TESTS_NAMESPACE_BEGIN

BEGIN_SUITE(ChannelEqualizer)

BEGIN_TEST(TestRGB24)

auto pSrcBitmap = IBitmap::Create(GetPathToTestFile("PPM/IMG_8970.ppm"));
auto pDstBitmap = ChannelEqualizer::AutoEqualize( pSrcBitmap );
EXPECT_TRUE(BitmapsAreEqual(GetPathToPattern("ChannelEqualizer/TestRGB24.ppm"), pDstBitmap));
END_TEST

BEGIN_TEST(TestRGB48 )

auto pSrcBitmap = IBitmap::Create( GetPathToTestFile( "PPM/sagittarius.ppm" ) );
auto pDstBitmap = ChannelEqualizer::AutoEqualize( pSrcBitmap );
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "ChannelEqualizer/TestRGB48.ppm" ), pDstBitmap ) );
END_TEST

BEGIN_TEST( TestGray8 )
auto pSrcBitmap = IBitmap::Create( GetPathToTestFile( "Tiff/Gray8.tiff" ) );
auto pDstBitmap = ChannelEqualizer::AutoEqualize( pSrcBitmap );
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "ChannelEqualizer/TestGray8.ppm" ), pDstBitmap ) );
END_TEST

BEGIN_TEST( TestGray16 )
auto pSrcBitmap = IBitmap::Create( GetPathToTestFile( "Tiff/Gray16.tiff" ) );
auto pDstBitmap = ChannelEqualizer::AutoEqualize( pSrcBitmap );
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "ChannelEqualizer/TestGray16.ppm" ), pDstBitmap ) );
END_TEST

BEGIN_TEST( TestNullArgs )

auto f = [] { ChannelEqualizer::AutoEqualize( nullptr ); };
ASSERT_THROWS( f, std::invalid_argument );
END_TEST

END_SUITE

ACMB_TESTS_NAMESPACE_END