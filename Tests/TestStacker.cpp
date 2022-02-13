#include "test.h"
#include "testtools.h"
#include "../Codecs/RAW/rawdecoder.h"
#include "../Registrator/stacker.h"
#include <filesystem>

BEGIN_SUITE(Stacker)

BEGIN_TEST(Stacker, BasicTest)

    std::vector<std::shared_ptr<ImageDecoder>> decoders
    {
      ImageDecoder::Create(GetPathToTestFile("PPM/IMG_4296.ppm")),
      ImageDecoder::Create(GetPathToTestFile("PPM/IMG_4314.ppm")),
      ImageDecoder::Create(GetPathToTestFile("PPM/IMG_4322.ppm"))
    };

    auto pStacker = std::make_shared<Stacker>(decoders);
    EXPECT_TRUE(BitmapsAreEqual(GetPathToPattern("Stacker/BasicTest.ppm"), pStacker->Stack(true)));

END_TEST

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
        ImageDecoder::Create(GetPathToTestFile("RAW/MilkyWayCR2/IMG_8945.CR2"))        
    };

    auto pStacker = std::make_shared<Stacker>(decoders);
    pStacker->Registrate(60, 5, 25);
    auto pStacked = pStacker->Stack(true);
    EXPECT_TRUE(BitmapsAreEqual(GetPathToPattern("Stacker/TestTwoPics.ppm"), pStacked));

END_TEST

BEGIN_TEST(Stacker, TestThreePics)

std::vector<std::shared_ptr<ImageDecoder>> decoders
{
    ImageDecoder::Create(GetPathToTestFile("RAW/MilkyWayCR2/IMG_8944.CR2")),
    ImageDecoder::Create(GetPathToTestFile("RAW/MilkyWayCR2/IMG_8945.CR2")),
    ImageDecoder::Create(GetPathToTestFile("RAW/MilkyWayCR2/IMG_8946.CR2"))
};

auto pStacker = std::make_shared<Stacker>(decoders);
pStacker->Registrate(60, 5, 25);
auto pStacked = pStacker->Stack(true);
EXPECT_TRUE(BitmapsAreEqual(GetPathToPattern("Stacker/TestThreePics.ppm"), pStacked));

END_TEST

BEGIN_TEST(Stacker, TestMilkyWay)

std::vector<std::shared_ptr<ImageDecoder>> decoders;
for (const auto& path : std::filesystem::directory_iterator(GetPathToTestFile("RAW/MilkyWayCR2/")))
{
    decoders.push_back(std::make_shared<RawDecoder>(false));
    decoders.back()->Attach(path.path().generic_string());
}

auto pStacker = std::make_shared<Stacker>(decoders);
pStacker->Registrate(15, 5, 25);
EXPECT_TRUE(BitmapsAreEqual(GetPathToPattern("Stacker/TestMilkyWay.ppm"), pStacker->Stack(true)));

END_TEST

END_SUITE
