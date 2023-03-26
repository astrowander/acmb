#include "test.h"
#include "testtools.h"
#include "../Core/pipeline.h"
#include "../Transforms/ChannelEqualizer.h"
#include "../Transforms/binningtransform.h"
#include "../Transforms/BitmapSubtractor.h"
#include "../Transforms/HaloRemovalTransform.h"
#include "../Transforms/ResizeTransform.h"
#include "../Transforms/converter.h"
#include "../Codecs/PPM/ppmdecoder.h"

ACMB_TESTS_NAMESPACE_BEGIN

BEGIN_SUITE( Pipeline )

BEGIN_TEST( TestNullArgs )
auto f1 = []
{
    Pipeline pipeline( nullptr );
};
ASSERT_THROWS( f1, std::invalid_argument );

auto f2 = []
{
    Pipeline pipeline;
    pipeline.Add( nullptr );
};
ASSERT_THROWS( f2, std::invalid_argument );
END_TEST

BEGIN_TEST( AddTransformToTheStart )
auto f = []
{
    Pipeline pipeline;
    pipeline.AddTransform<AutoChannelEqualizer>();
};
ASSERT_THROWS( f, std::runtime_error );
END_TEST

BEGIN_TEST( AddDecoderToTheMiddle )
auto f = []
{
    Pipeline pipeline ( std::make_shared<PpmDecoder>() );
    pipeline.Add( std::make_shared<PpmDecoder>() );
};
ASSERT_THROWS( f, std::runtime_error );
END_TEST

BEGIN_TEST( TestBinning )
    Pipeline pipeline( ImageDecoder::Create( GetPathToTestFile( "PPM/rgb24.ppm" ) ) );
    pipeline.AddTransform<BinningTransform>( Size{ 2, 2 } );
    auto pBitmap = pipeline.RunAndGetBitmap();
    EXPECT_EQ( 320, pBitmap->GetWidth() );
    EXPECT_EQ( 213, pBitmap->GetHeight() );
END_TEST

BEGIN_TEST( TestLongPipeline )
Pipeline pipeline( ImageDecoder::Create( GetPathToTestFile( "PPM/IMG_8970.ppm" ) ) );
pipeline.AddTransform<ResizeTransform>( { 5496, 3670 } );
pipeline.AddTransform<Converter>( PixelFormat::RGB48 );
pipeline.AddTransform<BitmapSubtractor>( IBitmap::Create( GetPathToTestFile( "DarkFrame/masterdark.ppm" ) ) );
pipeline.AddTransform<AutoChannelEqualizer>();
pipeline.AddTransform<AutoHaloRemoval>();
pipeline.AddTransform<Converter>( PixelFormat::Gray8 );
pipeline.AddTransform<BinningTransform>( Size{ 2, 2 } );

auto pBitmap = pipeline.RunAndGetBitmap();
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "Pipeline/TestLongPipeline.ppm" ), pBitmap ) );
END_TEST

END_SUITE

ACMB_TESTS_NAMESPACE_END
