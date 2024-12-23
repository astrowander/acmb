#include "test.h"
#include "testtools.h"
#include "../Transforms/BitmapHealer.h"
#include "../Transforms/converter.h"

ACMB_TESTS_NAMESPACE_BEGIN

#define TEST_PIXEL_FORMAT( fmt ) \
BEGIN_TEST ( Test##fmt ) \
auto pSrcBitmap = IBitmap::Create( GetPathToTestFile( "JPEG/IMG_4175.JPG" ) );\
pSrcBitmap = Converter::Convert( pSrcBitmap, PixelFormat::fmt ); \
std::vector <BitmapHealer::Patch > patches = { {.from = { 5104, 545 }, .to = { 4904, 435 }, .radius = 80, .gamma = 1.0f } };\
auto pDstBitmap = BitmapHealer::ApplyTransform( pSrcBitmap, patches );\
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "BitmapHealer/Test" #fmt ".ppm" ), pDstBitmap ) );\
END_TEST

BEGIN_SUITE( BitmapHealer )

TEST_PIXEL_FORMAT( RGB24 )
TEST_PIXEL_FORMAT( RGB48 )
TEST_PIXEL_FORMAT( Gray8 )
TEST_PIXEL_FORMAT( Gray16 )

END_SUITE

ACMB_TESTS_NAMESPACE_END