#include "test.h"
#include "testtools.h"
#include "../Codecs/MP4/Mp4Encoder.h"
#include "../Codecs/JPEG/JpegDecoder.h"

#include <filesystem>

ACMB_TESTS_NAMESPACE_BEGIN

BEGIN_SUITE( Mp4Encoder )

BEGIN_TEST( TestRGB24 )

    Mp4Encoder encoder( H264Preset::Medium );
    encoder.Attach( GetPathToPattern( "Mp4Encoder/TestRGB24.mp4" ) );
    encoder.SetFrameRate( 30 );
    JpegDecoder jpegDecoder;

    for ( auto entry : std::filesystem::directory_iterator( "D:/FILES/PHOTOS/timelapses/2019_07_kireevsk/night1/Resized/" ) )
    {
        if ( entry.is_directory() )
            continue;
        const auto path = entry.path();
        if ( !JpegDecoder::GetExtensions().contains( path.extension().string() ) )
             continue;

        jpegDecoder.Attach( path.string() );
        encoder.WriteBitmap( jpegDecoder.ReadBitmap() );
        jpegDecoder.Detach();
    }

    encoder.Detach();

END_TEST
END_SUITE
ACMB_TESTS_NAMESPACE_END