#include "test.h"
#include "testtools.h"
#include "../Core/pipeline.h"
#include "../Codecs/Raw/RawDecoder.h"
#include "../Registrator/stacker.h"
#include "../Transforms/BitmapSubtractor.h"
#include "../Transforms/BitmapDivisor.h"
#include <filesystem>

ACMB_TESTS_NAMESPACE_BEGIN

BEGIN_SUITE(Stacker)

BEGIN_TEST(TestStackingWithoutAlignment)

    std::vector<Pipeline> pipelines;
    for (const auto& path : std::filesystem::directory_iterator(GetPathToTestFile("RAW/TestStackingWithoutAlignment/")))
    {
        auto pDecoder = std::make_shared<RawDecoder>();
        pDecoder->Attach( path.path().generic_string() );
        pipelines.emplace_back( pDecoder );
    }

    auto pStacker = std::make_shared<Stacker>( *pipelines.front().GetFinalParams(), StackMode::LightNoAlign);
    pStacker->AddBitmaps( pipelines );
    EXPECT_TRUE(BitmapsAreEqual(GetPathToPattern("Stacker/TestStackingWithoutAlignment.ppm"), pStacker->GetResult()));

END_TEST

BEGIN_TEST( TestRgb24 )

std::vector<Pipeline> pipelines;
for ( const auto& path : std::filesystem::directory_iterator( GetPathToTestFile( "RAW/TestStackingWithoutAlignment/" ) ) )
{
    auto pDecoder = std::make_shared<RawDecoder>( PixelFormat::RGB24 );
    pDecoder->Attach( path.path().generic_string() );
    pipelines.emplace_back( pDecoder );
}

auto pStacker = std::make_shared<Stacker>( *pipelines.front().GetFinalParams(), StackMode::LightNoAlign );
pStacker->AddBitmaps( pipelines );

EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "Stacker/TestRgb24.ppm" ), pStacker->GetResult() ) );

END_TEST

BEGIN_TEST( TestGray8 )

std::vector<Pipeline> pipelines;
for ( const auto& path : std::filesystem::directory_iterator( GetPathToTestFile( "RAW/TestStackingWithoutAlignment/" ) ) )
{
    auto pDecoder = std::make_shared<RawDecoder>( PixelFormat::Gray8 );
    pDecoder->Attach( path.path().generic_string() );
    pipelines.emplace_back( pDecoder );
}

auto pStacker = std::make_shared<Stacker>( *pipelines.front().GetFinalParams(), StackMode::LightNoAlign );
pStacker->AddBitmaps( pipelines );
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "Stacker/TestGray8.ppm" ), pStacker->GetResult() ) );

END_TEST

BEGIN_TEST( TestGray16 )

std::vector<Pipeline> pipelines;
for ( const auto& path : std::filesystem::directory_iterator( GetPathToTestFile( "RAW/TestStackingWithoutAlignment/" ) ) )
{
    auto pDecoder = std::make_shared<RawDecoder>( PixelFormat::Gray16 );
    pDecoder->Attach( path.path().generic_string() );
    pipelines.emplace_back( pDecoder );
}

auto pStacker = std::make_shared<Stacker>( *pipelines.front().GetFinalParams(), StackMode::LightNoAlign );
pStacker->AddBitmaps( pipelines );
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "Stacker/TestGray16.ppm" ), pStacker->GetResult() ) );

END_TEST

BEGIN_TEST(TestTwoPics)

    std::vector<Pipeline> pipelines
    {
        { ImageDecoder::Create( GetPathToTestFile( "RAW/MilkyWayCR2/IMG_8944.CR2" ) ) },
        { ImageDecoder::Create( GetPathToTestFile( "RAW/MilkyWayCR2/IMG_8970.CR2" ) ) }
    };

    auto pStacker = std::make_shared<Stacker>( *pipelines.front().GetFinalParams(), StackMode::Light );
    pStacker->AddBitmaps( pipelines );
    auto pStacked = pStacker->GetResult();
    EXPECT_TRUE(BitmapsAreEqual(GetPathToPattern("Stacker/TestTwoPics.ppm"), pStacked));

END_TEST

BEGIN_TEST(TestEquatorialRegion)

std::vector<Pipeline> pipelines
{
    { ImageDecoder::Create( GetPathToTestFile( "RAW/Equator/IMG_9423.CR2" ) ) },
    { ImageDecoder::Create( GetPathToTestFile( "RAW/Equator/IMG_9442.CR2" ) ) }
};

auto pStacker = std::make_shared<Stacker>( *pipelines.front().GetFinalParams(), StackMode::Light );
pStacker->AddBitmaps( pipelines );
auto pStacked = pStacker->GetResult();
EXPECT_TRUE(BitmapsAreEqual(GetPathToPattern("Stacker/TestEquatorialRegion.ppm"), pStacked));

END_TEST

BEGIN_TEST(TestThreePics)

const std::vector<Pipeline> pipelines
{
    { ImageDecoder::Create( GetPathToTestFile( "RAW/MilkyWayCR2/IMG_8944.CR2" ) ) },
    { ImageDecoder::Create( GetPathToTestFile( "RAW/MilkyWayCR2/IMG_8945.CR2" ) ) },
    { ImageDecoder::Create( GetPathToTestFile( "RAW/MilkyWayCR2/IMG_8946.CR2" ) ) }
};

auto pStacker = std::make_shared<Stacker>( *pipelines.front().GetFinalParams(), StackMode::Light );
pStacker->AddBitmaps( pipelines );
auto pStacked = pStacker->GetResult();
EXPECT_TRUE(BitmapsAreEqual(GetPathToPattern("Stacker/TestThreePics.ppm"), pStacked));

END_TEST

BEGIN_TEST(TestMilkyWay)

auto pipelines = ImageDecoder::GetPipelinesFromDir( GetPathToTestFile( "RAW/MilkyWayCR2/" ) );
auto pStacker = std::make_shared<Stacker>( *pipelines.front().GetFinalParams(), StackMode::Light );
pStacker->AddBitmaps( pipelines );
EXPECT_TRUE(BitmapsAreEqual(GetPathToPattern("Stacker/TestMilkyWay.ppm"), pStacker->GetResult()));

END_TEST

BEGIN_TEST(TestFastStacking)

std::vector<Pipeline> pipelines
{
    { ImageDecoder::Create( GetPathToTestFile( "RAW/MilkyWayCR2/IMG_8944.CR2" ) ) },
    { ImageDecoder::Create( GetPathToTestFile( "RAW/MilkyWayCR2/IMG_8945.CR2" ) ) },
    { ImageDecoder::Create( GetPathToTestFile( "RAW/MilkyWayCR2/IMG_8946.CR2" ) ) }
};

auto pStacker = std::make_shared<Stacker>( *pipelines.front().GetFinalParams(), StackMode::Light );
pStacker->AddBitmaps( pipelines );
EXPECT_TRUE(BitmapsAreEqual(GetPathToPattern("Stacker/TestFastStacking.ppm"), pStacker->GetResult()));

END_TEST

BEGIN_TEST( StackWithDarks )

    auto darkPipelines = ImageDecoder::GetPipelinesFromDir( GetPathToTestFile( "RAW/StackWithDarks/Darks/" ) );
    auto pDarkStacker = std::make_shared<Stacker>( *darkPipelines.front().GetFinalParams(), StackMode::DarkOrFlat);
    pDarkStacker->AddBitmaps( darkPipelines );
    auto pDarkFrame = pDarkStacker->GetResult();
    pDarkStacker.reset();
    EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "Stacker/masterdark.ppm" ), pDarkFrame ) );

    auto pipelines = ImageDecoder::GetPipelinesFromDir( GetPathToTestFile( "RAW/StackWithDarks/Lights/" ) );    
    for ( auto& pipeline : pipelines )
    {        
        pipeline.AddTransform<BitmapSubtractor>( { pDarkFrame } );
    }

    auto pStacker = std::make_shared<Stacker>( *pipelines.front().GetFinalParams(), StackMode::Light );
    pStacker->AddBitmaps( pipelines );
    EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "Stacker/StackWithDarks.ppm" ), pStacker->GetResult() ) );

END_TEST

BEGIN_TEST( StackWithDarksAndFlats )

auto darkPipelines = ImageDecoder::GetPipelinesFromDir( GetPathToTestFile( "RAW/StackWithDarks/Darks/" ) );
auto pDarkStacker = std::make_shared<Stacker>( *darkPipelines.front().GetFinalParams(), StackMode::DarkOrFlat );
pDarkStacker->AddBitmaps( darkPipelines );
auto pDarkFrame = pDarkStacker->GetResult();

auto darkFlatPipelines = ImageDecoder::GetPipelinesFromDir( GetPathToTestFile( "RAW/StackWithDarks/DarkFlats/" ) );
auto pDarkFlatStacker = std::make_shared<Stacker>( *darkFlatPipelines.front().GetFinalParams(), StackMode::DarkOrFlat);
pDarkFlatStacker->AddBitmaps( darkFlatPipelines );
auto pDarkFlatFrame = pDarkStacker->GetResult();

auto flatPipelines = ImageDecoder::GetPipelinesFromDir( GetPathToTestFile( "RAW/StackWithDarks/Flats/" ) );
for ( auto& pipeline : flatPipelines )
{
    pipeline.AddTransform<BitmapSubtractor>( { pDarkFlatFrame } );
}

auto pFlatStacker = std::make_shared<Stacker>( *flatPipelines.front().GetFinalParams(), StackMode::DarkOrFlat );
pFlatStacker->AddBitmaps( flatPipelines );
auto pFlatFrame = pFlatStacker->GetResult();
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "Stacker/masterflat.ppm" ), pFlatFrame ) );

auto pipelines = ImageDecoder::GetPipelinesFromDir( GetPathToTestFile( "RAW/StackWithDarks/Lights/" ) );
for ( auto& pipeline : pipelines )
{
    pipeline.AddTransform<BitmapSubtractor>( { pDarkFrame } );
    pipeline.AddTransform<BitmapDivisor>( { .pDivisor = pFlatFrame } );
}

auto pStacker = std::make_shared<Stacker>( *pipelines.front().GetFinalParams(), StackMode::Light );
pStacker->AddBitmaps( pipelines );
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "Stacker/StackWithDarksAndFlats.ppm" ), pStacker->GetResult() ) );

END_TEST

BEGIN_TEST ( TestNullArgs )

auto f = []
{
    std::vector<Pipeline> pipelines{ { nullptr } };
    auto pStacker = std::make_shared<Stacker>( *pipelines.front().GetFinalParams(), StackMode::Light );
};

ASSERT_THROWS( f, std::invalid_argument );
END_TEST

BEGIN_TEST( TestStarTrails )
std::vector<Pipeline> pipelines
{
    { ImageDecoder::Create( GetPathToTestFile( "RAW/MilkyWayCR2/IMG_8944.CR2" ) ) },
    { ImageDecoder::Create( GetPathToTestFile( "RAW/MilkyWayCR2/IMG_8945.CR2" ) ) },
    { ImageDecoder::Create( GetPathToTestFile( "RAW/MilkyWayCR2/IMG_8946.CR2" ) ) }
};

auto pStacker = std::make_shared<Stacker>( *pipelines.front().GetFinalParams(), StackMode::StarTrails );
pStacker->AddBitmaps( pipelines );
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "Stacker/TestStarTrails.ppm" ), pStacker->GetResult() ) );
END_TEST

END_SUITE

ACMB_TESTS_NAMESPACE_END
