#include "test.h"
#include "testtools.h"
#include "../Transforms/binningtransform.h"

BEGIN_SUITE(BinningTransform)

BEGIN_TEST(BinningTransform, TestHugePicture)

EXPECT_TRUE(BitmapsAreEqual(GetPathToPattern("BinningTransform/Huge_2x2.ppm"), IBinningTransform<2, 2>::Create(IBitmap::Create(GetPathToTestFile("PPM/IMG_4314.ppm")))->RunAndGetBitmap()));

END_TEST

BEGIN_TEST(BinningTransform, TestRGB24)

auto pSrcBitmap = IBitmap::Create(GetPathToTestFile("PPM/rgb24.ppm"));
auto pTransform = IBinningTransform<2, 2>::Create(pSrcBitmap);
auto pDstBitmap = pTransform->RunAndGetBitmap();
EXPECT_TRUE(BitmapsAreEqual(GetPathToPattern("BinningTransform/RGB24_2x2.ppm"), pDstBitmap));

END_TEST

BEGIN_TEST(BinningTransform, Test3x3)

auto pSrcBitmap = IBitmap::Create(GetPathToTestFile("PPM/rgb24.ppm"));
auto pTransform = IBinningTransform<3, 3>::Create(pSrcBitmap);
auto pDstBitmap = pTransform->RunAndGetBitmap();
EXPECT_TRUE(BitmapsAreEqual(GetPathToPattern("BinningTransform/RGB24_3x3.ppm"), pDstBitmap));

END_TEST

END_SUITE
