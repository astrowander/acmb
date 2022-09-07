#include "test.h"
#include "testtools.h"
#include "../Codecs/TIFF/TiffDecoder.h"
#include "../Codecs/TIFF/TiffEncoder.h"
#include "../Codecs/PPM/PpmEncoder.h"
#include "../Core/bitmap.h"
#include <filesystem>
#include "stdio.h"


static bool TestPixelFormat(const std::string& pixelFormat)
{
    auto pRefBitmap = IBitmap::Create( GetPathToTestFile( std::string("TIFF/") +  pixelFormat + ".tiff" ) );
    TiffEncoder tiffEncoder;
    auto tempDir = std::filesystem::temp_directory_path();

    PpmEncoder ppmEncoder(PpmMode::Binary);
    std::string ppmFileName = tempDir.string() + pixelFormat + "_temp.ppm";
    ppmEncoder.Attach( ppmFileName );
    ppmEncoder.WriteBitmap( pRefBitmap );
    ppmEncoder.Detach();


    std::string tmpFileName = tempDir.string() + pixelFormat + "_temp.tif";

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

END_SUITE( TiffEncoder )