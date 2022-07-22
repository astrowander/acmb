#include "test.h"
#include "testtools.h"
#include "../Codecs/RAW/rawdecoder.h"
#include "../Registrator/stacker.h"
#include <filesystem>

BEGIN_SUITE(Stacker)

BEGIN_TEST(Stacker, TestStackingWithoutAlignment)

    std::vector<std::shared_ptr<ImageDecoder>> decoders;
    for (const auto& path : std::filesystem::directory_iterator(GetPathToTestFile("RAW/TestStackingWithoutAlignment/")))
    {
        decoders.push_back(std::make_shared<RawDecoder>(true));
        decoders.back()->Attach(path.path().generic_string());
    }

    auto pStacker = std::make_shared<Stacker>(decoders);
    EXPECT_TRUE(BitmapsAreEqual(GetPathToPattern("Stacker/TestStackingWithoutAlignment.ppm"), pStacker->Stack(false)));

END_TEST

BEGIN_TEST(Stacker, TestTwoPics)

    std::vector<std::shared_ptr<ImageDecoder>> decoders
    {
        ImageDecoder::Create(GetPathToTestFile("RAW/MilkyWayCR2/IMG_8944.CR2")),
        ImageDecoder::Create(GetPathToTestFile("RAW/MilkyWayCR2/IMG_8970.CR2"))        
    };

    auto pStacker = std::make_shared<Stacker>(decoders, false);
    pStacker->Registrate(25, 5, 25);
    auto pStacked = pStacker->Stack(true);
    EXPECT_TRUE(BitmapsAreEqual(GetPathToPattern("Stacker/TestTwoPics.ppm"), pStacked));

END_TEST

BEGIN_TEST(Stacker, TestEquatorialRegion)

std::vector<std::shared_ptr<ImageDecoder>> decoders
{
    ImageDecoder::Create(GetPathToTestFile("RAW/Equator/IMG_9423.CR2")),
    ImageDecoder::Create(GetPathToTestFile("RAW/Equator/IMG_9442.CR2"))
};

auto pStacker = std::make_shared<Stacker>(decoders, false);
pStacker->Registrate(25, 5, 25);
auto pStacked = pStacker->Stack(true);
EXPECT_TRUE(BitmapsAreEqual(GetPathToPattern("Stacker/TestEquatorialRegion.ppm"), pStacked));

END_TEST

BEGIN_TEST(Stacker, TestThreePics)

std::vector<std::shared_ptr<ImageDecoder>> decoders
{
    ImageDecoder::Create(GetPathToTestFile("RAW/MilkyWayCR2/IMG_8944.CR2")),
    ImageDecoder::Create(GetPathToTestFile("RAW/MilkyWayCR2/IMG_8945.CR2")),
    ImageDecoder::Create(GetPathToTestFile("RAW/MilkyWayCR2/IMG_8946.CR2"))
};

auto pStacker = std::make_shared<Stacker>(decoders);
pStacker->Registrate(25, 5, 25);
auto pStacked = pStacker->Stack(true);
EXPECT_TRUE(BitmapsAreEqual(GetPathToPattern("Stacker/TestThreePics.ppm"), pStacked));

END_TEST

BEGIN_TEST(Stacker, TestMilkyWay)

auto pStacker = std::make_shared<Stacker>( ImageDecoder::GetDecodersFromDir( GetPathToTestFile( "RAW/MilkyWayCR2/" ) ) );
EXPECT_TRUE(BitmapsAreEqual(GetPathToPattern("Stacker/TestMilkyWay.ppm"), pStacker->RegistrateAndStack(25, 5, 25)));

END_TEST

BEGIN_TEST(Stacker, TestFastStacking)

std::vector<std::shared_ptr<ImageDecoder>> decoders
{
    ImageDecoder::Create(GetPathToTestFile("RAW/MilkyWayCR2/IMG_8944.CR2")),
    ImageDecoder::Create(GetPathToTestFile("RAW/MilkyWayCR2/IMG_8945.CR2")),
    ImageDecoder::Create(GetPathToTestFile("RAW/MilkyWayCR2/IMG_8946.CR2"))
};

auto pStacker = std::make_shared<Stacker>(decoders);
EXPECT_TRUE(BitmapsAreEqual(GetPathToPattern("Stacker/TestFastStacking.ppm"), pStacker->RegistrateAndStack(25, 5, 25)));

END_TEST

BEGIN_TEST( Stacker, StackWithDarks )

    auto pDarkStacker = std::make_shared<Stacker>( ImageDecoder::GetDecodersFromDir( GetPathToTestFile( "RAW/StackWithDarks/Darks/" ) ) );
    auto pDarkFrame = pDarkStacker->Stack( false );
    pDarkStacker.reset();

    auto pStacker = std::make_shared<Stacker>( ImageDecoder::GetDecodersFromDir( GetPathToTestFile( "RAW/StackWithDarks/Lights/" ) ) );
    pStacker->SetDarkFrame( pDarkFrame );
    EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "Stacker/StackWithDarks.ppm" ), pStacker->RegistrateAndStack( 25, 5, 25 ) ) );

END_TEST

END_SUITE (Stacker)
