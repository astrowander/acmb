#include "test.h"
#include "testtools.h"
#include "./../Codecs/Raw/RawDecoder.h"
#include "./../Codecs/PPM/ppmencoder.h"

ACMB_TESTS_NAMESPACE_BEGIN

BEGIN_SUITE(RawDecoder)

BEGIN_TEST(TestAttach)

auto pDecoder = std::make_unique<RawDecoder>();
pDecoder->Attach(GetPathToTestFile( "RAW/MilkyWayCR2/IMG_8944.CR2" ));
EXPECT_EQ(PixelFormat::Gray16, pDecoder->GetPixelFormat());
EXPECT_EQ(5496, pDecoder->GetWidth());
EXPECT_EQ(3670, pDecoder->GetHeight());

END_TEST

BEGIN_TEST(TestReadBitmap)

auto pDecoder = std::make_unique<RawDecoder>();
pDecoder->Attach(GetPathToTestFile( "RAW/MilkyWayCR2/IMG_8944.CR2" ));
auto pBitmap = pDecoder->ReadBitmap();

EXPECT_TRUE(BitmapsAreEqual(GetPathToPattern("RawDecoder/IMG_8944.ppm"), pBitmap));
END_TEST

BEGIN_TEST( JustRead )

auto pDecoder = std::make_unique<RawDecoder>();
pDecoder->Attach( GetPathToTestFile( "RAW/MilkyWayCR2/IMG_8944.CR2" ) );
auto pBitmap = pDecoder->ReadBitmap();
END_TEST

BEGIN_TEST(TestDNG)

auto pDecoder = std::make_unique<RawDecoder>( RawSettings {.halfSize = true, .outputFormat = PixelFormat::RGB48 } );
pDecoder->Attach(GetPathToTestFile("RAW/IMG_20211020_190808.dng"));
EXPECT_EQ(PixelFormat::RGB48, pDecoder->GetPixelFormat());
EXPECT_EQ(4096, pDecoder->GetWidth());
EXPECT_EQ(3072, pDecoder->GetHeight());
auto pBitmap = pDecoder->ReadBitmap();
EXPECT_TRUE(BitmapsAreEqual(GetPathToPattern("RawDecoder/IMG_20211020_190808.ppm"), pBitmap));

END_TEST

BEGIN_TEST(TestEmptyFile)

auto f = []()
{
	auto pDecoder = std::make_unique<RawDecoder>();
	pDecoder->Attach(GetPathToTestFile("RAW/empty.CR2"));
};

ASSERT_THROWS(f, std::runtime_error);
END_TEST

BEGIN_TEST(TestCorruptedFile)

auto f = []()
{
	auto pDecoder = std::make_unique<RawDecoder>();
	pDecoder->Attach(GetPathToTestFile("RAW/corrupted.CR2"));
};

ASSERT_THROWS(f, std::runtime_error);
END_TEST

BEGIN_TEST( TestRGB24 )

auto pDecoder = std::make_unique<RawDecoder>( RawSettings {.halfSize = false, .outputFormat = PixelFormat::RGB24 });
pDecoder->Attach( GetPathToTestFile( "RAW/MilkyWayCR2/IMG_8944.CR2" ) );
EXPECT_EQ( PixelFormat::RGB24, pDecoder->GetPixelFormat() );
EXPECT_EQ( 5496, pDecoder->GetWidth() );
EXPECT_EQ( 3670, pDecoder->GetHeight() );

auto pBitmap = pDecoder->ReadBitmap();

EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "RawDecoder/TestRGB24.ppm" ), pBitmap ) );

END_TEST

BEGIN_TEST ( TestGray16 )

auto pDecoder = std::make_unique<RawDecoder>( RawSettings{ .halfSize = false, .outputFormat = PixelFormat::Gray16 } );
pDecoder->Attach( GetPathToTestFile( "RAW/MilkyWayCR2/IMG_8944.CR2" ) );
EXPECT_EQ( PixelFormat::Gray16, pDecoder->GetPixelFormat() );
EXPECT_EQ( 5496, pDecoder->GetWidth() );
EXPECT_EQ( 3670, pDecoder->GetHeight() );

auto pBitmap = pDecoder->ReadBitmap();
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "RawDecoder/TestGray16.pgm" ), pBitmap ) );
END_TEST

BEGIN_TEST( TestGray8 )

auto f = []
{
	auto pDecoder = std::make_unique<RawDecoder>( RawSettings{ .halfSize = false, .outputFormat = PixelFormat::Gray8 } );
};

ASSERT_THROWS( f, std::invalid_argument );
END_TEST

END_SUITE

ACMB_TESTS_NAMESPACE_END
