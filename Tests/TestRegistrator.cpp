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

BEGIN_TEST(Registrator, RegistrateHugePhoto)
auto pBitmap = IBitmap::Create(GetPathToTestFile("PPM/IMG_4314.ppm"));
auto stars = Registrator::Registrate(pBitmap, 50, 5, 25);
EXPECT_EQ(8971, stars.size());
EXPECT_EQ(299, stars[2].pixelCount);
EXPECT_NEAR(0.80, stars[2].luminance, 0.01);
EXPECT_EQ((Rect{1468, 2368, 15, 24}), stars[2].rect);
END_TEST

BEGIN_TEST(Registrator, TestVertical)
auto pBitmap = IBitmap::Create(GetPathToTestFile("PPM/vertical.ppm"));
auto stars = Registrator::Registrate(pBitmap, 40, 5, 25);
EXPECT_EQ(1654, stars.size());
EXPECT_EQ(227, stars[1].pixelCount);
EXPECT_NEAR(0.54, stars[1].luminance, 0.01);
EXPECT_EQ((Rect{143, 594, 16, 20}), stars[1].rect);
END_TEST

END_SUITE
