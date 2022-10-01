#include "test.h"
#include "testtools.h"
#include "../Transforms/converter.h"

#define PIXEL_FORMAT_TEST(fmt1, fmt2) \
BEGIN_TEST(Test ##fmt1 ## To ##fmt2)\
EXPECT_TRUE( ( BitmapsAreEqual( GetPathToPattern( "Converter/Test" #fmt1 "To" #fmt2 ".tif" ), Converter::Convert( IBitmap::Create( GetPathToTestFile( "TIFF/" #fmt1 ".tiff" ) ), PixelFormat::##fmt2 ) ) ) )\
END_TEST

ACMB_TESTS_NAMESPACE_BEGIN

BEGIN_SUITE(Converter)

PIXEL_FORMAT_TEST(RGB24, Gray8)
PIXEL_FORMAT_TEST( RGB24, Gray16 )
PIXEL_FORMAT_TEST( RGB24, RGB24 )
PIXEL_FORMAT_TEST( RGB24, RGB48 )

PIXEL_FORMAT_TEST( RGB48, Gray8 )
PIXEL_FORMAT_TEST( RGB48, Gray16 )
PIXEL_FORMAT_TEST( RGB48, RGB24 )
PIXEL_FORMAT_TEST( RGB48, RGB48 )

PIXEL_FORMAT_TEST( Gray8, Gray8 )
PIXEL_FORMAT_TEST( Gray8, Gray16 )
PIXEL_FORMAT_TEST( Gray8, RGB24 )
PIXEL_FORMAT_TEST( Gray8, RGB48 )

PIXEL_FORMAT_TEST( Gray16, Gray8 )
PIXEL_FORMAT_TEST( Gray16, Gray16 )
PIXEL_FORMAT_TEST( Gray16, RGB24 )
PIXEL_FORMAT_TEST( Gray16, RGB48 )

BEGIN_TEST(TestAstrophoto)
    EXPECT_TRUE((BitmapsAreEqual(GetPathToPattern("Converter/IMG_4030.ppm"), Converter::Convert(IBitmap::Create(GetPathToTestFile("PPM/IMG_4030.ppm")), PixelFormat::Gray16))));
END_TEST

BEGIN_TEST( TestNullArgs )
auto f = []
{
    Converter::Convert( nullptr, PixelFormat::Gray8 );
};
ASSERT_THROWS( f, std::invalid_argument );
END_TEST

END_SUITE

ACMB_TESTS_NAMESPACE_END