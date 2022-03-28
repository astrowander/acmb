#include "test.h"
#include "testtools.h"
#include "../Registrator/registrator.h"

BEGIN_SUITE(Registrator)

BEGIN_TEST(Registrator, BasicTest)

auto pRegistrator = std::make_unique<Registrator>(1, 1, 10);

pRegistrator->Registrate(IBitmap::Create(GetPathToTestFile("PPM/IMG_4030.ppm")));
auto stars = pRegistrator->GetStars()[0];

EXPECT_EQ(173, stars.size());
EXPECT_EQ(1.0, stars[0].luminance);
EXPECT_EQ(151, stars[0].pixelCount);
EXPECT_EQ((Rect{327, 229, 12, 20}), stars[0].rect);

END_TEST

BEGIN_TEST(Registrator, RegistrateHugePhoto)
auto pRegistrator = std::make_unique<Registrator>(1, 1, 50);
pRegistrator->Registrate(IBitmap::Create(GetPathToTestFile("PPM/IMG_8970.ppm")));
auto stars = pRegistrator->GetStars()[0];

EXPECT_EQ(16722, stars.size());
EXPECT_EQ(99, stars[2].pixelCount);
EXPECT_NEAR(0.79, stars[2].luminance, 0.01);
EXPECT_EQ((Rect{4028, 3513, 11, 12}), stars[2].rect);
END_TEST

BEGIN_TEST(Registrator, TestVertical)
auto pRegistrator = std::make_unique<Registrator>(1, 1, 40);
pRegistrator->Registrate(IBitmap::Create(GetPathToTestFile("PPM/vertical.ppm")));
auto stars = pRegistrator->GetStars()[0];

EXPECT_EQ(15352, stars.size());
EXPECT_EQ(250, stars[1].pixelCount);
EXPECT_NEAR(0.6, stars[1].luminance, 0.01);
EXPECT_EQ((Rect{3048, 2782, 19, 22}), stars[1].rect);
END_TEST

BEGIN_TEST(Registrator, TestMultipleTiles)
auto pRegistrator = std::make_unique<Registrator>(8, 6, 25);
pRegistrator->Registrate(IBitmap::Create(GetPathToTestFile("RAW/MilkyWayCR2/IMG_8944.CR2")));
auto stars = pRegistrator->GetStars();
EXPECT_EQ(48, stars.size());

uint32_t sum = 0;
for (auto& tileStars : stars)
{
	sum += tileStars.size();
}
EXPECT_EQ(9146, sum);
END_TEST

BEGIN_TEST(Registrator, Test1x1Bitmap)
auto pRegistrator = std::make_unique<Registrator>(1, 1, 25);
pRegistrator->Registrate(IBitmap::Create(1, 1, PixelFormat::RGB24));
EXPECT_EQ(0, pRegistrator->GetStars()[0].size());
END_TEST

BEGIN_TEST(Registrator, Test1x1BitmapMultipleTiles)
auto f = []()
{
	auto pRegistrator = std::make_unique<Registrator>(2, 2, 25);
	pRegistrator->Registrate(IBitmap::Create(1, 1, PixelFormat::RGB24));
};

ASSERT_THROWS(f, std::invalid_argument);

END_TEST

BEGIN_TEST(Registrator, TestBlackBitmap)
	auto pRegistrator = std::make_unique<Registrator>(2, 2, 25);
	pRegistrator->Registrate(IBitmap::Create(100, 100, PixelFormat::RGB24));
	EXPECT_EQ(0, pRegistrator->GetStars()[0].size());
END_TEST

END_SUITE
