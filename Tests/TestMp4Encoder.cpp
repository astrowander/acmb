#include "test.h"
#include "testtools.h"
#include "../Codecs/MP4/Mp4Encoder.h"
#include "../Codecs/JPEG/JpegDecoder.h"
#include "../Transforms/ResizeTransform.h"

#include <filesystem>

ACMB_TESTS_NAMESPACE_BEGIN

BEGIN_SUITE( Mp4Encoder )

BEGIN_TEST( TestRGB24 )

    Mp4Encoder encoder( H265Preset::VerySlow, H265Tune::ZeroLatency );
    encoder.Attach( "C:/ffmpeg/frames/output.h265");
    encoder.SetFrameRate( 30 );
    JpegDecoder jpegDecoder;
    
    int count = 0;
    for ( auto entry : std::filesystem::directory_iterator( "C:/ffmpeg/frames/" ) )
    {
        if ( entry.is_directory() )
            continue;
        const auto path = entry.path();
        if ( !JpegDecoder::GetExtensions().contains( path.extension().string() ) )
             continue;

        jpegDecoder.Attach( path.string() );
        auto bitmap = jpegDecoder.ReadBitmap();
        //auto pResize = ResizeTransform::Create( PixelFormat::RGB24, { .width = 256, .height = 144 } );
        //auto pResizedBitmap = ResizeTransform::Resize( bitmap, { .width = 720, .height = 480 } );
        //bitmap->Save( pResizedBitmap, GetPathToPattern( "Mp4Encoder/bitmap.ppm" ) );
        encoder.WriteBitmap( bitmap );
        jpegDecoder.Detach();
        std::cout << count++ << "frames encoded" << std::endl;
    }

    encoder.Detach();

END_TEST
END_SUITE
ACMB_TESTS_NAMESPACE_END