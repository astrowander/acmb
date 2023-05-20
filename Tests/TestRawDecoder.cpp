#include "test.h"
#include "testtools.h"
#include "./../Codecs/Raw/RawDecoder.h"
#include "./../Codecs/PPM/ppmencoder.h"
#include <fstream>
#include <thread>

ACMB_TESTS_NAMESPACE_BEGIN

BEGIN_SUITE(RawDecoder)

BEGIN_TEST(TestAttach)

auto pDecoder = std::make_unique<RawDecoder>();
pDecoder->Attach(GetPathToTestFile( "RAW/MilkyWayCR2/IMG_8944.CR2" ));
EXPECT_EQ(PixelFormat::Bayer16, pDecoder->GetPixelFormat());
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

auto pDecoder = std::make_unique<RawDecoder>( PixelFormat::RGB48 );
pDecoder->Attach(GetPathToTestFile("RAW/IMG_20211020_190808.dng"));
EXPECT_EQ(PixelFormat::RGB48, pDecoder->GetPixelFormat());
EXPECT_EQ(8192, pDecoder->GetWidth());
EXPECT_EQ(6144, pDecoder->GetHeight());
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

auto pDecoder = std::make_unique<RawDecoder>( PixelFormat::RGB24 );
pDecoder->Attach( GetPathToTestFile( "RAW/MilkyWayCR2/IMG_8944.CR2" ) );
EXPECT_EQ( PixelFormat::RGB24, pDecoder->GetPixelFormat() );
EXPECT_EQ( 5496, pDecoder->GetWidth() );
EXPECT_EQ( 3670, pDecoder->GetHeight() );

auto pBitmap = pDecoder->ReadBitmap();

EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "RawDecoder/TestRGB24.ppm" ), pBitmap ) );

END_TEST

BEGIN_TEST ( TestGray16 )

auto pDecoder = std::make_unique<RawDecoder>( PixelFormat::Gray16 );
pDecoder->Attach( GetPathToTestFile( "RAW/MilkyWayCR2/IMG_8944.CR2" ) );
EXPECT_EQ( PixelFormat::Gray16, pDecoder->GetPixelFormat() );
EXPECT_EQ( 5496, pDecoder->GetWidth() );
EXPECT_EQ( 3670, pDecoder->GetHeight() );

auto pBitmap = pDecoder->ReadBitmap();
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "RawDecoder/TestGray16.pgm" ), pBitmap ) );
END_TEST

BEGIN_TEST( TestGray8 )

auto pDecoder = std::make_unique<RawDecoder>( PixelFormat::Gray8 );
pDecoder->Attach( GetPathToTestFile( "RAW/MilkyWayCR2/IMG_8944.CR2" ) );
EXPECT_EQ( PixelFormat::Gray8, pDecoder->GetPixelFormat() );
EXPECT_EQ( 5496, pDecoder->GetWidth() );
EXPECT_EQ( 3670, pDecoder->GetHeight() );

auto pBitmap = pDecoder->ReadBitmap();
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "RawDecoder/TestGray8.pgm" ), pBitmap ) );
END_TEST

BEGIN_TEST (TestLensInfo)
auto pDecoder = std::make_unique<RawDecoder>();

pDecoder->Attach( GetPathToTestFile( "RAW/TestLensInfo/Canon_200_mm.CR2" ) );
EXPECT_EQ( "Canon", pDecoder->GetCameraSettings()->lensMakerName );
EXPECT_EQ( "Canon EF 200mm f/2.8L II USM", pDecoder->GetCameraSettings()->lensModelName );

pDecoder = std::make_unique<RawDecoder>();
pDecoder->Attach( GetPathToTestFile( "RAW/TestLensInfo/Canon_24_105_mm.CR2" ) );
EXPECT_EQ( "Canon", pDecoder->GetCameraSettings()->lensMakerName );
EXPECT_EQ( "Canon EF 24-105mm f/4L IS USM", pDecoder->GetCameraSettings()->lensModelName );

pDecoder = std::make_unique<RawDecoder>();
pDecoder->Attach( GetPathToTestFile( "RAW/TestLensInfo/Samyang_85_mm.CR2" ) );
EXPECT_EQ( "", pDecoder->GetCameraSettings()->lensMakerName );
EXPECT_EQ( "", pDecoder->GetCameraSettings()->lensModelName );


END_TEST

BEGIN_TEST (TestReadingFromStream)

std::ifstream is( GetPathToTestFile( "RAW/MilkyWayCR2/IMG_8944.CR2" ), std::ios_base::in | std::ios_base::binary );
if ( !is.is_open() )
    throw std::runtime_error( "unable to open input file" );

std::string buf;
is.seekg( 0, is.end );
const size_t length = is.tellg();
is.seekg( 0, is.beg );

buf.resize( length );
is.read( buf.data(), length );
auto pInStream = std::make_shared<std::istringstream>( buf );
std::shared_ptr<ImageDecoder> pDecoder;
pDecoder = std::make_unique<RawDecoder>( PixelFormat::RGB48 );
pDecoder->Attach( pInStream );

EXPECT_EQ( PixelFormat::RGB48, pDecoder->GetPixelFormat() );
EXPECT_EQ( 5496, pDecoder->GetWidth() );
EXPECT_EQ( 3670, pDecoder->GetHeight() );
auto pBitmap = pDecoder->ReadBitmap();
EXPECT_TRUE(BitmapsAreEqual(GetPathToPattern("RawDecoder/IMG_8944.ppm"), pBitmap));

END_TEST

END_SUITE

ACMB_TESTS_NAMESPACE_END
