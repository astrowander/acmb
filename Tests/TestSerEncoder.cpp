#include "test.h"
#include "testtools.h"
#include "../Codecs/Ser/SerEncoder.h"
#include "../Codecs/Ser/SerDecoder.h"
#include "../Transforms/converter.h"
ACMB_TESTS_NAMESPACE_BEGIN

template<PixelFormat pixelFormat>
bool TestPixelFormat()
{
    auto pDecoder = ImageDecoder::Create( GetPathToTestFile( "/SER/19_45_36.ser" ) );
    
    std::shared_ptr<std::stringstream> pStream( new std::stringstream );
    auto pEncoder = std::make_shared<SerEncoder>();
    pEncoder->Attach( pStream );

    for ( int i = 0; i < pDecoder->GetFrameCount(); ++i )
    {
        pEncoder->WriteBitmap( Converter::Convert( pDecoder->ReadBitmap(), pixelFormat ) );
    }

    pDecoder->Reattach();
    pEncoder->Detach();
    pStream->seekg( 0, std::ios::beg );

    auto pResultDecoder = std::make_shared<SerDecoder>();
    pResultDecoder->Attach( pStream );

    if ( pDecoder->GetFrameCount() != pResultDecoder->GetFrameCount() )
        return false;

    if ( pixelFormat != pResultDecoder->GetPixelFormat() )
        return false;

    if ( pDecoder->GetWidth() != pResultDecoder->GetWidth() )
        return false;

    if ( pDecoder->GetHeight() != pResultDecoder->GetHeight() )
        return false;

    for ( int i = 0; i < pDecoder->GetFrameCount(); ++i )
    {
        if ( !BitmapsAreEqual( Converter::Convert( pDecoder->ReadBitmap(), pixelFormat ), pResultDecoder->ReadBitmap() ) )
            return false;
    }
    return true;
}

BEGIN_SUITE( SerEncoder )

BEGIN_TEST( TestRGB24 )
EXPECT_TRUE( TestPixelFormat<PixelFormat::RGB24>() );
END_TEST

BEGIN_TEST( TestRGB48 )
EXPECT_TRUE( TestPixelFormat<PixelFormat::RGB48>() );
END_TEST

BEGIN_TEST( TestGray8 )
EXPECT_TRUE( TestPixelFormat<PixelFormat::Gray8>() );
END_TEST

BEGIN_TEST( TestGray16 )
EXPECT_TRUE( TestPixelFormat<PixelFormat::Gray16>() );
END_TEST

BEGIN_TEST( TestBayer16 )
EXPECT_TRUE( TestPixelFormat<PixelFormat::Gray16>() );
END_TEST

BEGIN_TEST( TestNullBitmap )
    SerEncoder encoder;
    std::shared_ptr<std::stringstream> pStream( new std::stringstream );
    encoder.Attach( pStream );
    encoder.WriteBitmap( nullptr );
    EXPECT_EQ( 0, encoder.GetTotalFrames() );
END_TEST

BEGIN_TEST( TestSizeMismatch )
auto f = []
{
    SerEncoder encoder;
    std::shared_ptr<std::stringstream> pStream( new std::stringstream );
    encoder.Attach( pStream );
    encoder.WriteBitmap( std::shared_ptr<Bitmap<PixelFormat::RGB24>>( new Bitmap<PixelFormat::RGB24>( 64, 64, 255 ) ) );
    encoder.WriteBitmap( std::shared_ptr<Bitmap<PixelFormat::RGB24>>( new Bitmap<PixelFormat::RGB24>( 65, 65, 255 ) ) );
};

ASSERT_THROWS( f, std::runtime_error );
END_TEST

BEGIN_TEST( TestFormatMismatch )
auto f = []
{
    SerEncoder encoder;
    std::shared_ptr<std::stringstream> pStream( new std::stringstream );
    encoder.Attach( pStream );
    encoder.WriteBitmap( std::shared_ptr<Bitmap<PixelFormat::RGB24>>( new Bitmap<PixelFormat::RGB24>( 64, 64, 255 ) ) );
    encoder.WriteBitmap( std::shared_ptr<Bitmap<PixelFormat::RGB48>>( new Bitmap<PixelFormat::RGB48>( 64, 64, 255 ) ) );
};

ASSERT_THROWS( f, std::runtime_error );
END_TEST


END_SUITE

ACMB_TESTS_NAMESPACE_END