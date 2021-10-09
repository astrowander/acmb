#include "test.h"
#include "testtools.h"
#include "Registrator/aligner.h"
#include "Registrator/registrator.h"
#include "Codecs/imagedecoder.h"

BEGIN_SUITE(Aligner)

BEGIN_TEST(Aligner, BasicTest)

std::vector<std::shared_ptr<ImageDecoder>> decoders
{
    ImageDecoder::Create(GetPathToTestFile("PPM/IMG_4314.ppm")),
    ImageDecoder::Create(GetPathToTestFile("PPM/IMG_4322.ppm"))
};

auto datasets = Aligner::Align(decoders);

std::cout << std::setw(8) <<  datasets[0]->transform.tx << std::setw(8) << datasets[0]->transform.ty << std::setw(40) << datasets[1]->transform.tx << std::setw(8) << datasets[1]->transform.ty << std::endl;
for (uint32_t i = 0; i < 20; ++ i)
{
    std::cout << std::setw(4) << i << std::setw(10) << datasets[0]->stars[i].luminance << std::setw(8) << datasets[0]->stars[i].rect.x << std::setw(8) << datasets[0]->stars[i].rect.y;
    std::cout << std::setw(20) << datasets[1]->stars[i].luminance << std::setw(8) << datasets[1]->stars[i].rect.x << std::setw(8) << datasets[1]->stars[i].rect.y << std::endl;

}

END_TEST
END_SUITE
