#include "test.h"
#include "testtools.h"
#include "Registrator/registrator.h"

BEGIN_SUITE(Registrator)

BEGIN_TEST(Registrator, BasicTest)

auto dataset = Registrator::Registrate(ImageDecoder::Create(GetPathToTestFile("PPM/IMG_4030.ppm")));
auto& stars = dataset->stars;
EXPECT_EQ(675, stars.size());
EXPECT_EQ(1.0, stars[0].luminance);
EXPECT_EQ(151, stars[0].pixelCount);
EXPECT_EQ((Rect{327, 229, 12, 20}), stars[0].rect);

END_TEST

BEGIN_TEST(Registrator, RegistrateHugePhoto)
auto dataset = Registrator::Registrate(ImageDecoder::Create(GetPathToTestFile("PPM/IMG_4314.ppm")), 50, 5, 25);
auto& stars = dataset->stars;
EXPECT_EQ(8971, stars.size());
EXPECT_EQ(196, stars[2].pixelCount);
EXPECT_NEAR(0.89, stars[2].luminance, 0.01);
EXPECT_EQ((Rect{1336, 1782, 11, 20}), stars[2].rect);
END_TEST

BEGIN_TEST(Registrator, TestVertical)
auto dataset = Registrator::Registrate(ImageDecoder::Create(GetPathToTestFile("PPM/vertical.ppm")), 40, 5, 25);
auto& stars = dataset->stars;
EXPECT_EQ(1654, stars.size());
EXPECT_EQ(115, stars[1].pixelCount);
EXPECT_NEAR(0.46, stars[1].luminance, 0.01);
EXPECT_EQ((Rect{648, 129, 11, 11}), stars[1].rect);
END_TEST

END_SUITE
