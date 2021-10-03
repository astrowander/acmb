#include "test.h"
#include "testtools.h"
#include "Codecs/PPM/ppmdecoder.h"
#include "Codecs/PPM/ppmencoder.h"
#include "Core/bitmap.h"

BEGIN_SUITE( PpmEncoder )

template<PixelFormat T, PpmMode mode>
static bool TestPixelFormat()
{
    auto pBitmap = std::make_shared<Bitmap<PixelFormat::Gray8>>(100, 100, ARGB32Color::White);
    auto openMode =  (mode == PpmMode::Binary) ? std::ios_base::out : std::ios_base::out | std::ios_base::binary;

    auto pOutStream = std::make_unique<std::ostringstream>(openMode);
    auto pEncoder = std::make_shared<PpmEncoder>(mode);
    pEncoder->Attach(std::move(pOutStream));
    pEncoder->WriteBitmap(pBitmap);

    openMode =  (mode == PpmMode::Binary) ? std::ios_base::in : std::ios_base::in | std::ios_base::binary;
    auto buf = static_cast<std::ostringstream*>(pEncoder->Detach().get())->str();
    auto pInStream = std::make_unique<std::istringstream>(buf, openMode);
    auto pDecoder = std::make_shared<PpmDecoder>();
    pDecoder->Attach(std::move(pInStream));

    return BitmapsAreEqual(pBitmap, pDecoder->GetBitmap());
}

BEGIN_TEST(PpmEncoder, TestGray8)

   EXPECT_TRUE((TestPixelFormat<PixelFormat::Gray8,PpmMode::Binary>()));
   EXPECT_TRUE((TestPixelFormat<PixelFormat::Gray8,PpmMode::Text>()));

END_TEST

BEGIN_TEST(PpmEncoder, TestGray16)

    EXPECT_TRUE((TestPixelFormat<PixelFormat::Gray16, PpmMode::Binary>()));
    EXPECT_TRUE((TestPixelFormat<PixelFormat::Gray16, PpmMode::Text>()));

END_TEST

BEGIN_TEST(PpmEncoder, TestRgb24)

    EXPECT_TRUE((TestPixelFormat<PixelFormat::RGB24, PpmMode::Binary>()));
    EXPECT_TRUE((TestPixelFormat<PixelFormat::RGB24, PpmMode::Text>()));

END_TEST

BEGIN_TEST(PpmEncoder, TestRgb48)

    EXPECT_TRUE((TestPixelFormat<PixelFormat::RGB48, PpmMode::Binary>()));
    EXPECT_TRUE((TestPixelFormat<PixelFormat::RGB48, PpmMode::Text>()));

END_TEST

END_SUITE
