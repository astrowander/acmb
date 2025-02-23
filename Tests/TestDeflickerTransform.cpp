#include "test.h"
#include "testtools.h"
#include "../Core/camerasettings.h"
#include "../Transforms/DebayerTransform.h"
#include "../Transforms/DeflickerTransform.h"
#include "../Codecs/Raw/RawDecoder.h"

ACMB_TESTS_NAMESPACE_BEGIN

BEGIN_SUITE( DeflickerTransform )

BEGIN_TEST( TestRGB48 )

std::vector<IBitmapPtr> bitmaps;
std::vector<std::string> fileNames;
for ( const auto& path : std::filesystem::directory_iterator( GetPathToTestFile( "RAW/Deflickering/" ) ) )
{
    auto pDecoder = std::make_shared<RawDecoder>( PixelFormat::RGB48 );
    pDecoder->Attach( path.path().generic_string() );
    bitmaps.push_back( pDecoder->ReadBitmap() );
    fileNames.push_back( path.path().filename().generic_string() );
}

DeflickerTransform::Settings settings;
settings.bitmaps = bitmaps;
settings.iterations = 3;
DeflickerTransform::Deflicker( settings );
for ( size_t i = 0; i < bitmaps.size(); i++ )
{
    auto pBitmap = bitmaps[i];
    EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "DeflickerTransform/" + fileNames[i] + ".ppm" ), pBitmap));
}
END_TEST

END_SUITE

ACMB_TESTS_NAMESPACE_END