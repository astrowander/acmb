#include "test.h"
#include "testtools.h"
#include "Registrator/aligner.h"
#include "Registrator/registrator.h"
#include "Codecs/imagedecoder.h"

BEGIN_SUITE(Aligner)

BEGIN_TEST(Aligner, BasicTest)

auto pRefBitmap = IBitmap::Create(GetPathToTestFile("PPM/IMG_4314.ppm"));
auto pRefDataset = Registrator::Registrate(pRefBitmap);
pRefBitmap.reset();

auto pTargetBitmap = IBitmap::Create(GetPathToTestFile("PPM/IMG_4322.ppm"));
auto pTargetDataset = Registrator::Registrate(pTargetBitmap);
pTargetBitmap.reset();

Aligner::Align(pRefDataset, pTargetDataset);

std::cout << std::setw(8) <<  pRefDataset->transform.tx << std::setw(8) << pRefDataset->transform.ty << std::setw(40) << pTargetDataset->transform.tx << std::setw(8) << pTargetDataset->transform.ty << std::endl;
for (uint32_t i = 0; i < 20; ++ i)
{
    std::cout << std::setw(4) << i << std::setw(10) << pRefDataset->stars[i].luminance << std::setw(12) << pRefDataset->stars[i].center.x << std::setw(12) << pRefDataset->stars[i].center.y;
    std::cout << std::setw(20) << pTargetDataset->stars[i].luminance << std::setw(12) << pTargetDataset->stars[i].center.x << std::setw(12) << pTargetDataset->stars[i].center.y << std::endl;

}

END_TEST
END_SUITE
