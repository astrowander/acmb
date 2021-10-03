#include "test.h"
#include "testtools.h"
#include "Registrator/registrator.h"

BEGIN_SUITE(Registrator)

BEGIN_TEST(Registrator, BasicTest)

auto pBitmap = IBitmap::Create(GetPathToTestFile("PPM/IMG_4030.ppm"));
auto stars = Registrator::Registrate(pBitmap);
std::cout << stars.size() << std::endl;

END_TEST
END_SUITE
