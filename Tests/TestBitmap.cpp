#include "test.h"
#include "testtools.h"
#include "../Core/bitmap.h"

BEGIN_SUITE(Bitmap)

BEGIN_TEST(Bitmap, TestGray8)

    auto pBitmap = std::make_unique<Bitmap<PixelFormat::Gray8>>(15, 20, ARGB32Color::Gray);
    EXPECT_EQ(127, pBitmap->GetChannel(0, 0, 0));

    EXPECT_EQ(300, pBitmap->GetByteSize());
    EXPECT_EQ(15, pBitmap->GetWidth());
    EXPECT_EQ(20, pBitmap->GetHeight());
    EXPECT_EQ(PixelFormat::Gray8, pBitmap->GetPixelFormat());

END_TEST

BEGIN_TEST(Bitmap, TestGray16)

    auto pBitmap = std::make_unique<Bitmap<PixelFormat::Gray16>>(15, 20, ARGB64Color::Gray);
    EXPECT_EQ(0x7FFF, pBitmap->GetChannel(0, 0, 0));

    EXPECT_EQ(600, pBitmap->GetByteSize());
    EXPECT_EQ(15, pBitmap->GetWidth());
    EXPECT_EQ(20, pBitmap->GetHeight());
    EXPECT_EQ(PixelFormat::Gray16, pBitmap->GetPixelFormat());

END_TEST

BEGIN_TEST(Bitmap, TestRgb48)
    auto pBitmap = std::make_unique<Bitmap<PixelFormat::RGB48>>(15, 20, ARGB64Color::Green);
    EXPECT_EQ(0x0000, pBitmap->GetChannel(0, 0, 0));
    EXPECT_EQ(0xFFFF, pBitmap->GetChannel(0, 0, 1));
    EXPECT_EQ(0x0000, pBitmap->GetChannel(0, 0, 2));

    EXPECT_EQ(1800, pBitmap->GetByteSize());
    EXPECT_EQ(15, pBitmap->GetWidth());
    EXPECT_EQ(20, pBitmap->GetHeight());
    EXPECT_EQ(PixelFormat::RGB48, pBitmap->GetPixelFormat());
END_TEST

BEGIN_TEST(Bitmap, TestRgb24)

    auto pBitmap = std::make_unique<Bitmap<PixelFormat::RGB24>>(15, 20, ARGB32Color::Red);
    EXPECT_EQ(0xFF, pBitmap->GetChannel(0, 0, 0));
    EXPECT_EQ(0x00, pBitmap->GetChannel(0, 0, 1));
    EXPECT_EQ(0x00, pBitmap->GetChannel(0, 0, 2));

    EXPECT_EQ(900, pBitmap->GetByteSize());
    EXPECT_EQ(15, pBitmap->GetWidth());
    EXPECT_EQ(20, pBitmap->GetHeight());
    EXPECT_EQ(PixelFormat::RGB24, pBitmap->GetPixelFormat());
END_TEST

BEGIN_TEST(Bitmap, TestInterpolation)

    auto pBitmap = IBitmap::Create(GetPathToTestFile("PPM/binary.ppm"));

    EXPECT_NEAR(28.125, pBitmap->GetInterpolatedChannel(0.5, 0.5, 0), 0.01);
    EXPECT_NEAR(147.51, pBitmap->GetInterpolatedChannel(1.75, 2.25, 2), 0.01);
    EXPECT_NEAR(189.263, pBitmap->GetInterpolatedChannel(2.85, 2.96, 1), 0.01);
END_TEST

BEGIN_TEST(Bitmap, TestZeroSize)
    auto f = []() {auto pBitmap = std::make_unique<Bitmap<PixelFormat::RGB24>>(0, 0, ARGB32Color::Red); };
    ASSERT_THROWS(f, std::invalid_argument);
END_TEST

BEGIN_TEST(Bitmap, TooLarge)
    auto f = []() {auto pBitmap = std::make_unique<Bitmap<PixelFormat::RGB24>>(-1, -1, ARGB32Color::Red); };
    ASSERT_THROWS(f, std::invalid_argument);
END_TEST

BEGIN_TEST(Bitmap, Test1x1Bitmap)
auto pBitmap = std::make_unique<Bitmap<PixelFormat::RGB24>>(1, 1, ARGB32Color::Red);
EXPECT_EQ(0xFF, pBitmap->GetChannel(0, 0, 0));
EXPECT_EQ(0x00, pBitmap->GetChannel(0, 0, 1));
EXPECT_EQ(0x00, pBitmap->GetChannel(0, 0, 2));
END_TEST
END_SUITE
