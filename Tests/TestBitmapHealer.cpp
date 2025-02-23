#include "test.h"
#include "testtools.h"
#include "../Transforms/BitmapHealer.h"
#include "../Transforms/CropTransform.h"
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

BEGIN_TEST( TestTransparency )

auto pSrcBitmap = IBitmap::Create( 500, 500, IColor::MakeRGB24( NamedColor32::White ) );
pSrcBitmap = CropTransform::CropAndFill( pSrcBitmap, { -250, 0, 500, 500 }, IColor::MakeRGB24( NamedColor32::Blue ) );
std::vector <BitmapHealer::Patch > patches = { {.from = { 350, 250 }, .to = { 150, 250 }, .radius = 25, .gamma = 1.0f } };
pSrcBitmap = BitmapHealer::ApplyTransform( pSrcBitmap, patches );
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "BitmapHealer/TestTransparency.ppm" ), pSrcBitmap ) );

END_TEST

END_SUITE

ACMB_TESTS_NAMESPACE_END