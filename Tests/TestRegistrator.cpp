#include "test.h"
#include "testtools.h"
#include "Registrator/registrator.h"

BEGIN_SUITE(Registrator)

BEGIN_TEST(Registrator, BasicTest)

auto pBitmap = IBitmap::Create(GetPathToTestFile("PPM/IMG_4030.ppm"));
auto stars = Registrator::Registrate(pBitmap);
EXPECT_EQ(675, stars.size());
EXPECT_EQ(1.0, stars[0].luminance);
EXPECT_EQ(151, stars[0].pixelCount);
EXPECT_EQ((Rect{327, 229, 12, 20}), stars[0].rect);

END_TEST
END_SUITE
