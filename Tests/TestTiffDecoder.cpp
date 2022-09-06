#include "test.h"
#include "testtools.h"
#include "../Codecs/TIFF/TiffDecoder.h"
#include "../Core/bitmap.h"

BEGIN_SUITE( TiffDecoder )

BEGIN_TEST( TestGray8 )

auto pDecoder = std::make_unique<TiffDecoder>();
pDecoder->Attach( GetPathToTestFile( "TIFF/Gray8.tiff" ) );
EXPECT_EQ( PixelFormat::Gray8, pDecoder->GetPixelFormat() );
EXPECT_EQ( 1280, pDecoder->GetWidth() );
EXPECT_EQ( 960, pDecoder->GetHeight() );
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "TiffDecoder/TestGray8.pgm" ), pDecoder->ReadBitmap() ) );

END_TEST

BEGIN_TEST( TestGray16 )

auto pDecoder = std::make_unique<TiffDecoder>();
pDecoder->Attach( GetPathToTestFile( "TIFF/Gray16.tiff" ) );
EXPECT_EQ( PixelFormat::Gray16, pDecoder->GetPixelFormat() );
EXPECT_EQ( 1280, pDecoder->GetWidth() );
EXPECT_EQ( 960, pDecoder->GetHeight() );
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "TiffDecoder/TestGray16.pgm" ), pDecoder->ReadBitmap() ) );

END_TEST

BEGIN_TEST( TestRGB24 )

auto pDecoder = std::make_unique<TiffDecoder>();
pDecoder->Attach( GetPathToTestFile( "TIFF/RGB24.tiff" ) );
EXPECT_EQ( PixelFormat::RGB24, pDecoder->GetPixelFormat() );
EXPECT_EQ( 1280, pDecoder->GetWidth() );
EXPECT_EQ( 960, pDecoder->GetHeight() );
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "TiffDecoder/TestRGB24.ppm" ), pDecoder->ReadBitmap() ) );

END_TEST


BEGIN_TEST( TestRGB48 )

auto pDecoder = std::make_unique<TiffDecoder>();
pDecoder->Attach( GetPathToTestFile( "TIFF/RGB48.tiff" ) );
EXPECT_EQ( PixelFormat::RGB48, pDecoder->GetPixelFormat() );
EXPECT_EQ( 1280, pDecoder->GetWidth() );
EXPECT_EQ( 960, pDecoder->GetHeight() );
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "TiffDecoder/TestRGB48.ppm" ), pDecoder->ReadBitmap() ) );

END_TEST

BEGIN_TEST( TestEmptyFile )
auto f = [] ()
{
    auto pDecoder = std::make_unique<TiffDecoder>();
    pDecoder->Attach( GetPathToTestFile( "TIFF/empty.tiff" ) );
};
ASSERT_THROWS( f, std::runtime_error );
END_TEST

BEGIN_TEST( TestCorruptedFile )
auto f = [] ()
{
    auto pDecoder = std::make_unique<TiffDecoder>();
    pDecoder->Attach( GetPathToTestFile( "TIFF/corrupted.tiff" ) );
};
ASSERT_THROWS( f, std::runtime_error );
END_TEST

BEGIN_TEST( ReadTwice )

auto pDecoder = std::make_unique<TiffDecoder>();
pDecoder->Attach( GetPathToTestFile( "Tiff/RGB24.tiff" ) );
auto pFirstBitmap = pDecoder->ReadBitmap();
auto pSecondBitmap = pDecoder->ReadBitmap();
EXPECT_TRUE( BitmapsAreEqual( pFirstBitmap, pSecondBitmap ) );

END_TEST

END_SUITE( TiffDecoder )