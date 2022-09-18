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
END_SUITE (ChannelEqualizer)

ACMB_TESTS_NAMESPACE_END