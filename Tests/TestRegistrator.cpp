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
pRegistrator->Registrate(IBitmap::Create(GetPathToTestFile("PPM/IMG_4314.ppm")));
auto stars = pRegistrator->GetStars()[0];

EXPECT_EQ(8971, stars.size());
EXPECT_EQ(196, stars[2].pixelCount);
EXPECT_NEAR(0.89, stars[2].luminance, 0.01);
EXPECT_EQ((Rect{1336, 1782, 11, 20}), stars[2].rect);
EXPECT_EQ(21, stars.size());
END_TEST

BEGIN_TEST(Registrator, TestVertical)
auto pRegistrator = std::make_unique<Registrator>(1, 1, 40);
pRegistrator->Registrate(IBitmap::Create(GetPathToTestFile("PPM/vertical.ppm")));
auto stars = pRegistrator->GetStars()[0];

EXPECT_EQ(1654, stars.size());
EXPECT_EQ(115, stars[1].pixelCount);
EXPECT_NEAR(0.46, stars[1].luminance, 0.01);
EXPECT_EQ((Rect{648, 129, 11, 11}), stars[1].rect);
END_TEST

BEGIN_TEST(Registrator, TestMultipleTiles)
std::cout << "start" << std::endl;
auto pRegistrator = std::make_unique<Registrator>(8, 6, 25);
pRegistrator->Registrate(IBitmap::Create(GetPathToTestFile("RAW/MilkyWayCR2/IMG_8944.CR2")));
auto stars = pRegistrator->GetStars();
EXPECT_EQ(48, stars.size());

uint32_t sum = 0;
for (auto& tileStars : stars)
{
	sum += tileStars.size();
}
std::cout << sum << " stars are found" << std::endl;
std::cout << "done" << std::endl;
END_TEST

END_SUITE
