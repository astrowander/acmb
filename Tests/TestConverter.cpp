#include "test.h"
#include "testtools.h"
#include "../Transforms/converter.h"

BEGIN_SUITE(Converter)

BEGIN_TEST(TestRgb24ToGray8)
    EXPECT_TRUE((BitmapsAreEqual(GetPathToPattern("Converter/TestRgb24ToGray8.ppm"), BaseConverter::Convert(IBitmap::Create(GetPathToTestFile("PPM/rgb24.ppm")), PixelFormat::Gray8))));
END_TEST

BEGIN_TEST(TestRgb48ToGray16)
    EXPECT_TRUE((BitmapsAreEqual(GetPathToPattern("Converter/TestRgb48ToGray16.ppm"), BaseConverter::Convert(IBitmap::Create(GetPathToTestFile("PPM/rgb48.ppm")), PixelFormat::Gray16))));
END_TEST

BEGIN_TEST(TestAstrophoto)
    EXPECT_TRUE((BitmapsAreEqual(GetPathToPattern("Converter/IMG_4030.ppm"), BaseConverter::Convert(IBitmap::Create(GetPathToTestFile("PPM/IMG_4030.ppm")), PixelFormat::Gray16))));
END_TEST

END_SUITE (Converter)
