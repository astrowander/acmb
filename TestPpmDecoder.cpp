#include "test.h"
#include "ppmdecoder.h"
#include "bitmap.h"
#include <cstring>

BEGIN_SUITE( PpmDecoder )

BEGIN_TEST(PpmDecoder, TestGray8)
    auto pDecoder = std::make_unique<PpmDecoder>();
    pDecoder->Attach("./../TestFiles/PPM/gray8.pgm");
    EXPECT_EQ(PixelFormat::Gray8, pDecoder->GetPixelFormat());
    EXPECT_EQ(500, pDecoder->GetWidth());
    EXPECT_EQ(500, pDecoder->GetHeight());
END_TEST

BEGIN_TEST(PpmDecoder, TestGray16)
    auto pDecoder = std::make_unique<PpmDecoder>();
    pDecoder->Attach("./../TestFiles/PPM/gray16.pgm");
    EXPECT_EQ(PixelFormat::Gray16, pDecoder->GetPixelFormat());
    EXPECT_EQ(500, pDecoder->GetWidth());
    EXPECT_EQ(500, pDecoder->GetHeight());
END_TEST

BEGIN_TEST(PpmDecoder, TestRgb24)
    auto pDecoder = std::make_unique<PpmDecoder>();
    pDecoder->Attach("./../TestFiles/PPM/rgb24.ppm");
    EXPECT_EQ(PixelFormat::RGB24, pDecoder->GetPixelFormat());
    EXPECT_EQ(640, pDecoder->GetWidth());
    EXPECT_EQ(426, pDecoder->GetHeight());
END_TEST

BEGIN_TEST(PpmDecoder, TestRgb48)
    auto pDecoder = std::make_unique<PpmDecoder>();
    pDecoder->Attach("./../TestFiles/PPM/rgb48.ppm");
    EXPECT_EQ(PixelFormat::RGB48, pDecoder->GetPixelFormat());
    EXPECT_EQ(500, pDecoder->GetWidth());
    EXPECT_EQ(500, pDecoder->GetHeight());
END_TEST

BEGIN_TEST(PpmDecoder, TestPlain)
    auto pDecoder = std::make_unique<PpmDecoder>();

    std::string fileNames[2] = { "./../TestFiles/PPM/plain.ppm", "./../TestFiles/PPM/binary.ppm" };

    for (auto& fileName : fileNames)
    {
        pDecoder->Attach(fileName);
        EXPECT_EQ(PixelFormat::RGB24, pDecoder->GetPixelFormat());
        EXPECT_EQ(4, pDecoder->GetWidth());
        EXPECT_EQ(4, pDecoder->GetHeight());

        auto pBitmap = pDecoder->GetBitmap();
        EXPECT_EQ(PixelFormat::RGB24, pBitmap->GetPixelFormat());
        EXPECT_EQ(4, pBitmap->GetWidth());
        EXPECT_EQ(4, pBitmap->GetHeight());

        const std::vector<uint8_t> expectedData
        {
            0, 0, 0,      100, 0, 0,      0, 0, 0,        255, 0, 255,
            0, 0, 0,      0, 255, 175,    0, 0, 0,        0, 0, 0,
            0, 0, 0,      0, 0, 0,        0, 15, 175,     0, 0, 0,
            255, 0, 255,  0, 0, 0, 0,     0, 0, 255,      255, 255
        };

        EXPECT_EQ(0, memcmp(std::static_pointer_cast<Bitmap<PixelFormat::RGB24>>(pBitmap)->GetPlanarScanline(0), expectedData.data(), expectedData.size()));
    }
END_TEST

END_SUITE
