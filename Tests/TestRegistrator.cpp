#include "test.h"
#include "testtools.h"
#include "../Registrator/registrator.h"

ACMB_TESTS_NAMESPACE_BEGIN

BEGIN_SUITE(Registrator)

BEGIN_TEST(BasicTest)

auto pRegistrator = std::make_unique<Registrator>(10);

pRegistrator->Registrate(IBitmap::Create(GetPathToTestFile("PPM/IMG_4030.ppm")));
auto stars = pRegistrator->GetStars()[0];

EXPECT_EQ(173, stars.size());
EXPECT_EQ(1.0, stars[0].luminance);
EXPECT_EQ(151, stars[0].pixelCount);
EXPECT_EQ((Rect{327, 229, 12, 20}), stars[0].rect);

END_TEST

BEGIN_TEST(RegistrateHugePhoto)
auto pRegistrator = std::make_unique<Registrator>(50);
pRegistrator->Registrate(IBitmap::Create(GetPathToTestFile("PPM/IMG_8970.ppm")));
auto stars = pRegistrator->GetStars()[0];

EXPECT_EQ(588, stars.size());
EXPECT_EQ(127, stars[2].pixelCount);
EXPECT_NEAR(0.32, stars[2].luminance, 0.01);
EXPECT_EQ((Rect{254, 586, 6, 14}), stars[2].rect);
END_TEST

BEGIN_TEST(TestVertical)
auto pRegistrator = std::make_unique<Registrator>(40);
pRegistrator->Registrate(IBitmap::Create(GetPathToTestFile("PPM/vertical.ppm")));
auto stars = pRegistrator->GetStars()[0];

EXPECT_EQ(439, stars.size());
EXPECT_EQ(97, stars[1].pixelCount);
EXPECT_NEAR(0.83, stars[1].luminance, 0.01);
EXPECT_EQ((Rect{518, 115, 9, 17}), stars[1].rect);
END_TEST

BEGIN_TEST(TestMultipleTiles)
auto pRegistrator = std::make_unique<Registrator>(25);
auto pBitmap = IBitmap::Create( GetPathToTestFile( "RAW/MilkyWayCR2/IMG_8944.CR2" ), PixelFormat::RGB48 );
pRegistrator->Registrate( pBitmap );
auto stars = pRegistrator->GetStars();
EXPECT_EQ(54, stars.size());

uint32_t sum = 0;
for (auto& tileStars : stars)
{
	sum += tileStars.size();
}
EXPECT_EQ(15544, sum);
END_TEST

BEGIN_TEST(Test1x1Bitmap)
auto pRegistrator = std::make_unique<Registrator>(25);
pRegistrator->Registrate(IBitmap::Create(1, 1, PixelFormat::RGB24));
EXPECT_EQ(0, pRegistrator->GetStars()[0].size());
END_TEST

BEGIN_TEST(Test1x1BitmapMultipleTiles)
auto f = []()
{
	auto pRegistrator = std::make_unique<Registrator>(25);
	pRegistrator->Registrate(IBitmap::Create(1, 1, PixelFormat::RGB24));
};

ASSERT_THROWS(f, std::invalid_argument);

END_TEST

BEGIN_TEST(TestBlackBitmap)
	auto pRegistrator = std::make_unique<Registrator>(25);
	pRegistrator->Registrate(IBitmap::Create(100, 100, PixelFormat::RGB24));
	EXPECT_EQ(0, pRegistrator->GetStars()[0].size());
END_TEST

BEGIN_TEST(TestGrayscale)
auto pRegistrator = std::make_unique<Registrator>( 25 );
auto pBitmap = IBitmap::Create( GetPathToTestFile( "TIFF/m65.tif" ) );
auto pBitmapCopy = pBitmap->Clone();
pRegistrator->Registrate( pBitmap );
EXPECT_TRUE( BitmapsAreEqual( pBitmap, pBitmapCopy ) );
END_TEST

BEGIN_TEST( TestNullArgs )
auto f = []
{
	auto pRegistrator = std::make_unique<Registrator>( 25 );
	pRegistrator->Registrate( nullptr );
};
ASSERT_THROWS( f, std::invalid_argument );
END_TEST

END_SUITE

ACMB_TESTS_NAMESPACE_END
