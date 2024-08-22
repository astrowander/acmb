#include "test.h"
#include "testtools.h"
#include "../Codecs/Y4M/Y4MEncoder.h"
#include "../Codecs/JPEG/JpegDecoder.h"
#include "../Transforms/ResizeTransform.h"

#include <sstream>
#include <fstream>
#include <filesystem>

ACMB_TESTS_NAMESPACE_BEGIN

BEGIN_SUITE( Y4MEncoder )

BEGIN_TEST( TestRGB24 )
    Y4MEncoder encoder;
    std::shared_ptr<std::stringstream> pStream( new std::stringstream );
    encoder.Attach( pStream );
    encoder.SetFrameRate( 25 );

    for ( int i = 0; i < 25; ++i )
    {
        const uint8_t l = uint8_t( std::clamp( float( i ) / 24.0f * 255.0f, 0.0f, 255.0f ) );
        encoder.WriteBitmap( std::shared_ptr<Bitmap<PixelFormat::RGB24>>( new Bitmap<PixelFormat::RGB24>( 64, 64, IColor::MakeRGB24( l, l, l ) ) ) );
    }

    for ( int i = 0; i < 25; ++i )
    {
        const uint8_t l = uint8_t( 255 - std::clamp( float( i ) / 24.0f * 255.0f, 0.0f, 255.0f ) );
        encoder.WriteBitmap( std::shared_ptr<Bitmap<PixelFormat::RGB24>>( new Bitmap<PixelFormat::RGB24>( 64, 64, IColor::MakeRGB24( l, l, l ) ) ) );
    }

    encoder.Detach();
    const auto encoded = pStream->str();

    const std::string dirPath = GetPathToPattern( "Y4MEncoder" );
    if ( !std::filesystem::exists( dirPath ) )
        std::filesystem::create_directory( dirPath );
    const std::string filePath = dirPath + "/rgb24.y4m";
    if ( !std::filesystem::exists( filePath ) )
    {
        std::ofstream fOut( filePath, std::ios::binary );
        fOut.write( encoded.data(), encoded.size() );
    }

    std::ifstream f( filePath, std::ios::binary );
    f.seekg( 0, std::ios::end );
    const auto length = f.tellg();
    f.seekg( 0, std::ios::beg );
    std::string expected( length, '\0' );
    f.read( &expected[0], length );
    EXPECT_EQ( expected, encoded );
END_TEST

BEGIN_TEST( TestUnsupportedPixelFormat )
    auto f = []
    {
        Y4MEncoder encoder;
        std::shared_ptr<std::stringstream> pStream( new std::stringstream );
        encoder.Attach( pStream );
        encoder.WriteBitmap( std::shared_ptr<Bitmap<PixelFormat::Gray8>>( new Bitmap<PixelFormat::Gray8>( 64, 64, IColor::MakeGray8( 255 ) ) ) );
    };

    ASSERT_THROWS( f, std::invalid_argument );
END_TEST

BEGIN_TEST( TestNullBitmap )
auto f = []
{
    Y4MEncoder encoder;
    std::shared_ptr<std::stringstream> pStream( new std::stringstream );
    encoder.Attach( pStream );
    encoder.WriteBitmap( nullptr );
};

ASSERT_THROWS( f, std::invalid_argument );
END_TEST

BEGIN_TEST( TestSizeMismatch )
auto f = []
{
    Y4MEncoder encoder;
    std::shared_ptr<std::stringstream> pStream( new std::stringstream );
    encoder.Attach( pStream );
    encoder.WriteBitmap( std::shared_ptr<Bitmap<PixelFormat::RGB24>>( new Bitmap<PixelFormat::RGB24>( 64, 64, IColor::MakeRGB24( 255, 255, 255 ) ) ) );
    encoder.WriteBitmap( std::shared_ptr<Bitmap<PixelFormat::RGB24>>( new Bitmap<PixelFormat::RGB24>( 65, 65, IColor::MakeRGB24( 255, 255, 255 ) ) ) );
};

ASSERT_THROWS( f, std::runtime_error );
END_TEST

END_SUITE
ACMB_TESTS_NAMESPACE_END
