#include "test.h"
#include "Core/bitmap.h"

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

END_SUITE
