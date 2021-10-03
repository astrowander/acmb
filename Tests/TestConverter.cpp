#include "test.h"
#include "testtools.h"
#include "Transforms/converter.h"

BEGIN_SUITE(Converter)

BEGIN_TEST(Converter, TestRgb24ToGray8)
    EXPECT_TRUE((BitmapsAreEqual("./../Patterns/Converter/TestRgb24ToGray8.ppm", Convert(IBitmap::Create("./../TestFiles/PPM/rgb24.ppm"), PixelFormat::Gray8))));
END_TEST

BEGIN_TEST(Converter, TestRgb48ToGray16)
    EXPECT_TRUE((BitmapsAreEqual("./../Patterns/Converter/TestRgb48ToGray16.ppm", Convert(IBitmap::Create("./../TestFiles/PPM/rgb48.ppm"), PixelFormat::Gray16))));
END_TEST

END_SUITE
