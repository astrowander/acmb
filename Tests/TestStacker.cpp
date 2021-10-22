#include "test.h"
#include "testtools.h"
#include "../Codecs/imagedecoder.h"
#include "../Registrator/stacker.h"

BEGIN_SUITE(Stacker)

BEGIN_TEST(Stacker, BasicTest)

    std::vector<std::shared_ptr<ImageDecoder>> decoders
    {
      ImageDecoder::Create(GetPathToTestFile("PPM/IMG_4296.ppm")),
      ImageDecoder::Create(GetPathToTestFile("PPM/IMG_4314.ppm")),
      ImageDecoder::Create(GetPathToTestFile("PPM/IMG_4322.ppm"))
    };

    auto pStacker = std::make_shared<Stacker>(decoders);
    EXPECT_TRUE(BitmapsAreEqual(GetPathToPattern("Stacker/BasicTest.ppm"), pStacker->Stack()));

END_TEST

END_SUITE
