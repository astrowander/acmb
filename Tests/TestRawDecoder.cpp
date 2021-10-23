#include "test.h"
#include "testtools.h"
#include "./../Codecs/Raw/RawDecoder.h"
#include "./../Codecs/PPM/PpmEncoder.h"
#include "./../Transforms/binningtransform.h"

BEGIN_SUITE(RawDecoder)

BEGIN_TEST(RawDecoder, TestAttach)

auto pDecoder = std::make_unique<RawDecoder>();
pDecoder->Attach(GetPathToTestFile("RAW/IMG_8899.CR2"));
EXPECT_EQ(PixelFormat::RGB24, pDecoder->GetPixelFormat());
EXPECT_EQ(5496, pDecoder->GetWidth());
EXPECT_EQ(3670, pDecoder->GetHeight());

END_TEST

BEGIN_TEST(RawDecoder, TestReadBitmap)

auto pDecoder = std::make_unique<RawDecoder>();
pDecoder->Attach(GetPathToTestFile("RAW/IMG_8899.CR2"));
auto pBitmap = pDecoder->ReadBitmap();

EXPECT_TRUE(BitmapsAreEqual(GetPathToPattern("RawDecoder/IMG_8899.ppm"), pBitmap));
END_TEST

BEGIN_TEST(RawDecoder, TestDNG)

auto pDecoder = std::make_unique<RawDecoder>();
pDecoder->Attach(GetPathToTestFile("RAW/IMG_20211020_190808.dng"));
EXPECT_EQ(PixelFormat::RGB24, pDecoder->GetPixelFormat());
EXPECT_EQ(8192, pDecoder->GetWidth());
EXPECT_EQ(6144, pDecoder->GetHeight());
auto pBitmap = pDecoder->ReadBitmap();
auto pBinningTransform = IBinningTransform<3,3>::Create(pBitmap);
pBitmap = pBinningTransform->RunAndGetBitmap();
EXPECT_TRUE(BitmapsAreEqual(GetPathToPattern("RawDecoder/IMG_20211020_190808.ppm"), pBitmap));

END_TEST

END_SUITE