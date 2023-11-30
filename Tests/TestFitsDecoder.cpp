#include "test.h"
#include "testtools.h"
#include "../Codecs/Fits/FitsDecoder.h"
#include "../Core/bitmap.h"

ACMB_TESTS_NAMESPACE_BEGIN

BEGIN_SUITE(FitsDecoder)

BEGIN_TEST( TestRGB48 )

auto pDecoder = std::make_unique<FitsDecoder>();
pDecoder->Attach( GetPathToTestFile( "FITS/rgb48.fit" ) );
EXPECT_EQ( PixelFormat::RGB48, pDecoder->GetPixelFormat() );
EXPECT_EQ( 5496, pDecoder->GetWidth() );
EXPECT_EQ( 3670, pDecoder->GetHeight() );
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "FitsDecoder/TestRGB48.ppm" ), pDecoder->ReadBitmap() ) );

END_TEST

BEGIN_TEST( TestGray16 )

auto pDecoder = std::make_unique<FitsDecoder>();
pDecoder->Attach( GetPathToTestFile( "FITS/gray16.fit" ) );
EXPECT_EQ( PixelFormat::Gray16, pDecoder->GetPixelFormat() );
EXPECT_EQ( 5496, pDecoder->GetWidth() );
EXPECT_EQ( 3670, pDecoder->GetHeight() );
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "FitsDecoder/TestGray16.ppm" ), pDecoder->ReadBitmap() ) );

END_TEST

BEGIN_TEST( TestOfficialSample )

auto pDecoder = std::make_unique<FitsDecoder>();
pDecoder->Attach( GetPathToTestFile( "FITS/UITfuv2582gc.fits" ) );
EXPECT_EQ( PixelFormat::Gray16, pDecoder->GetPixelFormat() );
EXPECT_EQ( 512, pDecoder->GetWidth() );
EXPECT_EQ( 512, pDecoder->GetHeight() );
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "FitsDecoder/UITfuv2582gc.ppm" ), pDecoder->ReadBitmap() ) );

END_TEST

BEGIN_TEST( TestEmptyFile )
auto f = [] ()
{
    auto pDecoder = std::make_unique<FitsDecoder>();
    pDecoder->Attach( GetPathToTestFile( "FITS/empty.fit" ) );
};
ASSERT_THROWS( f, std::runtime_error );
END_TEST

BEGIN_TEST( TestCorruptedFile )
auto f = [] ()
{
    auto pDecoder = std::make_unique<FitsDecoder>();
    pDecoder->Attach( GetPathToTestFile( "FITS/corrupted.fit" ) );
};
ASSERT_THROWS( f, std::runtime_error );
END_TEST

BEGIN_TEST( ReadTwice )

auto pDecoder = std::make_unique<FitsDecoder>();
pDecoder->Attach( GetPathToTestFile( "FITS/UITfuv2582gc.fits" ) );
auto pFirstBitmap = pDecoder->ReadBitmap();
auto pSecondBitmap = pDecoder->ReadBitmap();
EXPECT_TRUE( BitmapsAreEqual( pFirstBitmap, pSecondBitmap ) );

END_TEST

END_SUITE

ACMB_TESTS_NAMESPACE_END