#include "test.h"
#include "testtools.h"
#include "../Transforms/binningtransform.h"

#define TEST_PIXEL_FORMAT( fmt ) \
BEGIN_TEST( Test ##fmt ) \
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "BinningTransform/Test" #fmt ".tiff"), BinningTransform::Create(IBitmap::Create(GetPathToTestFile("TIFF/" #fmt ".tiff")), {2, 2})->RunAndGetBitmap()));\
END_TEST

ACMB_TESTS_NAMESPACE_BEGIN

BEGIN_SUITE(BinningTransform)

TEST_PIXEL_FORMAT( RGB24 )
TEST_PIXEL_FORMAT( RGB48 )
TEST_PIXEL_FORMAT( Gray8 )
TEST_PIXEL_FORMAT( Gray16 )

BEGIN_TEST(TestHugePicture)

EXPECT_TRUE(BitmapsAreEqual(GetPathToPattern("BinningTransform/Huge_2x2.ppm"), BinningTransform::Create(IBitmap::Create(GetPathToTestFile("PPM/IMG_8970.ppm")), { 2, 2 })->RunAndGetBitmap()));

END_TEST

BEGIN_TEST(Test3x3)

auto pSrcBitmap = IBitmap::Create(GetPathToTestFile("PPM/rgb24.ppm"));
auto pTransform = BinningTransform::Create(pSrcBitmap, {3, 3});
auto pDstBitmap = pTransform->RunAndGetBitmap();
EXPECT_TRUE(BitmapsAreEqual(GetPathToPattern("BinningTransform/RGB24_3x3.ppm"), pDstBitmap));

END_TEST

BEGIN_TEST(Test3x2)

auto pSrcBitmap = IBitmap::Create(GetPathToTestFile("PPM/rgb24.ppm"));
auto pTransform = BinningTransform::Create(pSrcBitmap, { 3, 2 });
auto pDstBitmap = pTransform->RunAndGetBitmap();
EXPECT_TRUE(BitmapsAreEqual(GetPathToPattern("BinningTransform/RGB24_3x2.ppm"), pDstBitmap));

END_TEST

BEGIN_TEST(Test1x1Bitmap)

auto f = []()
{
	auto pSrcBitmap = IBitmap::Create(1, 1, PixelFormat::RGB24);
	auto pTransform = BinningTransform::Create(pSrcBitmap, { 3, 2 });
	auto pDstBitmap = pTransform->RunAndGetBitmap();
};

ASSERT_THROWS(f, std::invalid_argument);

END_TEST

BEGIN_TEST(TestZeroSize)
	auto f = []() {BinningTransform::Create(IBitmap::Create(100, 100, PixelFormat::Gray8), { 3, 3 }); };
	ASSERT_THROWS(f, std::invalid_argument);
END_TEST

BEGIN_TEST( TestNullArg )
auto f = [] (){	BinningTransform::Create( nullptr, { 3, 3 } ); };
ASSERT_THROWS( f, std::invalid_argument );
END_TEST

END_SUITE

ACMB_TESTS_NAMESPACE_END