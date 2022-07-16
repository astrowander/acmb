#include "test.h"
#include "testtools.h"

#include "../Transforms/ChannelEqualizer.h"

BEGIN_SUITE(ChannelEqualizer)

BEGIN_TEST(ChannelEqualizer, TestRGB48)

auto pSrcBitmap = IBitmap::Create(GetPathToPattern("Stacker/TestMilkyWay.ppm"));
auto pChannelEqualizer = BaseChannelEqualizer::Create(pSrcBitmap, { 0.905, 1, 1.612 });
auto pDstBitmap = pChannelEqualizer->RunAndGetBitmap();
EXPECT_TRUE(BitmapsAreEqual(GetPathToPattern("ChannelEqualizer/TestRGB48.ppm"), pDstBitmap));
END_TEST
 END_SUITE (ChannelEqualizer)