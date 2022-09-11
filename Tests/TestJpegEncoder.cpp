#include "test.h"
#include "testtools.h"
#include "../Codecs/TIFF/TiffDecoder.h"
#include "../Codecs/Jpeg/JpegEncoder.h"
#include "../Core/bitmap.h"
#include <filesystem>
#include "stdio.h"

BEGIN_SUITE ( JpegEncoder )

BEGIN_TEST (TestRGB24)

auto pBitmap = IBitmap::Create( GetPathToTestFile( std::string( "TIFF/RGB24.tiff" ) ) );
JpegEncoder jpegEncoder;
jpegEncoder.Attach( GetPathToPattern( "JpegEncoder/TestRGB24.jpeg" ) );
jpegEncoder.WriteBitmap( pBitmap );
jpegEncoder.Detach();

END_TEST

BEGIN_TEST( TestGray8 )

auto pBitmap = IBitmap::Create( GetPathToTestFile( std::string( "TIFF/Gray8.tiff" ) ) );
JpegEncoder jpegEncoder;
jpegEncoder.Attach( GetPathToPattern( "JpegEncoder/TestGray8.jpeg" ) );
jpegEncoder.WriteBitmap( pBitmap );
jpegEncoder.Detach();

END_TEST

BEGIN_TEST( TestExtended )
    auto f = [] ()
    {
        auto pBitmap = IBitmap::Create( 100, 100, PixelFormat::RGB48 );
        JpegEncoder jpegEncoder;
        jpegEncoder.Attach( GetPathToPattern( "JpegEncoder/TestExtended.jpeg" ) );
    };

    ASSERT_THROWS( f, std::invalid_argument );

    END_TEST

BEGIN_TEST( TestNullBitmap )
    auto f = [] ()
    {
        IBitmapPtr pBitmap = nullptr;
        JpegEncoder jpegEncoder;
        jpegEncoder.Attach( GetPathToPattern( "JpegEncoder/TestNullBitmap.jpeg" ) );
        jpegEncoder.WriteBitmap( nullptr );
    };

    ASSERT_THROWS( f, std::invalid_argument );
END_TEST

END_SUITE( JpegEncoder )