#include "test.h"
#include "testtools.h"
#include "../Core/pipeline.h"
#include "../Codecs/Raw/RawDecoder.h"
#include "../Registrator/stacker.h"
#include <filesystem>

ACMB_TESTS_NAMESPACE_BEGIN

BEGIN_SUITE(Stacker)

BEGIN_TEST(TestStackingWithoutAlignment)

    std::vector<Pipeline> pipelines;
    for (const auto& path : std::filesystem::directory_iterator(GetPathToTestFile("RAW/TestStackingWithoutAlignment/")))
    {
        auto pDecoder = std::make_shared<RawDecoder>( RawSettings{ .halfSize = true, .extendedFormat = true } );
        pDecoder->Attach( path.path().generic_string() );
        pipelines.emplace_back( pDecoder );
    }

    auto pStacker = std::make_shared<Stacker>(pipelines);
    EXPECT_TRUE(BitmapsAreEqual(GetPathToPattern("Stacker/TestStackingWithoutAlignment.ppm"), pStacker->Stack()));

END_TEST

BEGIN_TEST(TestTwoPics)

    std::vector<Pipeline> pipelines
    {
        { ImageDecoder::Create( GetPathToTestFile( "RAW/MilkyWayCR2/IMG_8944.CR2" ) ) },
        { ImageDecoder::Create( GetPathToTestFile( "RAW/MilkyWayCR2/IMG_8970.CR2" ) ) }
    };

    auto pStacker = std::make_shared<Stacker>( pipelines);
    pStacker->Registrate();
    auto pStacked = pStacker->Stack();
    EXPECT_TRUE(BitmapsAreEqual(GetPathToPattern("Stacker/TestTwoPics.ppm"), pStacked));

END_TEST

BEGIN_TEST(TestEquatorialRegion)

std::vector<Pipeline> pipelines
{
    { ImageDecoder::Create( GetPathToTestFile( "RAW/Equator/IMG_9423.CR2" ) ) },
    { ImageDecoder::Create( GetPathToTestFile( "RAW/Equator/IMG_9442.CR2" ) ) }
};

auto pStacker = std::make_shared<Stacker>( pipelines);
pStacker->Registrate();
auto pStacked = pStacker->Stack();
EXPECT_TRUE(BitmapsAreEqual(GetPathToPattern("Stacker/TestEquatorialRegion.ppm"), pStacked));

END_TEST

BEGIN_TEST(TestThreePics)

std::vector<Pipeline> pipelines
{
    { ImageDecoder::Create( GetPathToTestFile( "RAW/MilkyWayCR2/IMG_8944.CR2" ) ) },
    { ImageDecoder::Create( GetPathToTestFile( "RAW/MilkyWayCR2/IMG_8945.CR2" ) ) },
    { ImageDecoder::Create( GetPathToTestFile( "RAW/MilkyWayCR2/IMG_8946.CR2" ) ) }
};

auto pStacker = std::make_shared<Stacker>( pipelines );
pStacker->Registrate();
auto pStacked = pStacker->Stack();
EXPECT_TRUE(BitmapsAreEqual(GetPathToPattern("Stacker/TestThreePics.ppm"), pStacked));

END_TEST

BEGIN_TEST(TestMilkyWay)

auto pStacker = std::make_shared<Stacker>( ImageDecoder::GetPipelinesFromDir( GetPathToTestFile( "RAW/MilkyWayCR2/" ) ) );
EXPECT_TRUE(BitmapsAreEqual(GetPathToPattern("Stacker/TestMilkyWay.ppm"), pStacker->RegistrateAndStack()));

END_TEST

BEGIN_TEST(TestFastStacking)

std::vector<Pipeline> pipelines
{
    { ImageDecoder::Create( GetPathToTestFile( "RAW/MilkyWayCR2/IMG_8944.CR2" ) ) },
    { ImageDecoder::Create( GetPathToTestFile( "RAW/MilkyWayCR2/IMG_8945.CR2" ) ) },
    { ImageDecoder::Create( GetPathToTestFile( "RAW/MilkyWayCR2/IMG_8946.CR2" ) ) }
};

auto pStacker = std::make_shared<Stacker>( pipelines );
EXPECT_TRUE(BitmapsAreEqual(GetPathToPattern("Stacker/TestFastStacking.ppm"), pStacker->RegistrateAndStack()));

END_TEST

BEGIN_TEST( StackWithDarks )

    auto pDarkStacker = std::make_shared<Stacker>( ImageDecoder::GetPipelinesFromDir( GetPathToTestFile( "RAW/StackWithDarks/Darks/" ) ) );
    auto pDarkFrame = pDarkStacker->Stack();
    pDarkStacker.reset();

    auto pStacker = std::make_shared<Stacker>( ImageDecoder::GetPipelinesFromDir( GetPathToTestFile( "RAW/StackWithDarks/Lights/" ) ) );
    pStacker->SetDarkFrame( pDarkFrame );
    EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "Stacker/StackWithDarks.ppm" ), pStacker->RegistrateAndStack() ) );

END_TEST

BEGIN_TEST ( TestNullArgs )

auto f = []
{
    std::vector<Pipeline> pipelines{ { nullptr } };
    auto pStacker = std::make_shared<Stacker>( pipelines );
};

ASSERT_THROWS( f, std::invalid_argument );
END_TEST

END_SUITE

ACMB_TESTS_NAMESPACE_END