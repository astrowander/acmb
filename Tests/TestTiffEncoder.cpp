#include "test.h"
#include "testtools.h"
#include "../Codecs/Tiff/TiffEncoder.h"
#include "../Codecs/PPM/ppmencoder.h"
#include "../Core/bitmap.h"
#include <filesystem>
#include <fstream>
#include <sstream>
#include "stdio.h"

ACMB_TESTS_NAMESPACE_BEGIN

static bool TestPixelFormat(const std::string& pixelFormat)
{
    auto pRefBitmap = IBitmap::Create( GetPathToTestFile( std::string("TIFF/") +  pixelFormat + ".tiff" ) );
    TiffEncoder tiffEncoder;
    auto tempDir = GetPathToTestFile("/tmp/");

    PpmEncoder ppmEncoder(PpmMode::Binary);
    std::string ppmFileName = tempDir + pixelFormat + "_temp.ppm";
    ppmEncoder.Attach( ppmFileName );
    ppmEncoder.WriteBitmap( pRefBitmap );
    ppmEncoder.Detach();


    std::string tmpFileName = tempDir + pixelFormat + "_temp.tif";

    tiffEncoder.Attach( tmpFileName );
    tiffEncoder.WriteBitmap( pRefBitmap );
    tiffEncoder.Detach();

    auto pTargetBitmap = IBitmap::Create( tmpFileName );
    return BitmapsAreEqual( pRefBitmap, pTargetBitmap );
}

BEGIN_SUITE( TiffEncoder )

BEGIN_TEST(TestGray8)

EXPECT_TRUE( TestPixelFormat( "Gray8" ) );

END_TEST

BEGIN_TEST( TestGray16 )

EXPECT_TRUE( TestPixelFormat( "Gray16" ) );

END_TEST

BEGIN_TEST( TestRGB24 )

EXPECT_TRUE( TestPixelFormat( "RGB24" ) );

END_TEST

BEGIN_TEST( TestRGB48 )

EXPECT_TRUE( TestPixelFormat( "RGB48" ) );

END_TEST

BEGIN_TEST( TestNullBitmap )
auto f = [] ()
{
    IBitmapPtr pBitmap = nullptr;
    TiffEncoder tiffEncoder;
    tiffEncoder.Attach( GetPathToPattern( "TiffEncoder/TestNullBitmap.tiff" ) );
    tiffEncoder.WriteBitmap( nullptr );
};

ASSERT_THROWS( f, std::invalid_argument );
END_TEST

BEGIN_TEST( TestWritingToStream )

auto pStream = std::make_shared<std::ostringstream>();
auto pRefBitmap = IBitmap::Create( GetPathToTestFile( std::string( "TIFF/RGB48.tiff" ) ) );

TiffEncoder tiffEncoder;
tiffEncoder.Attach( pStream );
tiffEncoder.WriteBitmap( pRefBitmap );
const auto str = pStream->str();
EXPECT_EQ( 7374351, str.size() );

END_TEST

END_SUITE

ACMB_TESTS_NAMESPACE_END
