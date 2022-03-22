#include "Tests/TestBitmap.cpp"
#include "Tests/TestPpmDecoder.cpp"
#include "Tests/TestPpmEncoder.cpp"
#include "Tests/TestConverter.cpp"
#include "Tests/TestRegistrator.cpp"
#include "Tests/TestFastAligner.cpp"
#include "Tests/TestBinningTransform.cpp"
#include "Tests/TestRawDecoder.cpp"
#include "Tests/TestStacker.cpp"
#include "Tests/TestNewton2D.cpp"
#include "Tests/TestStarTrekTransform.cpp"
#include <filesystem>
#include <thread>

int main()
{
    std::cout << std::filesystem::current_path() << std::endl;
    unsigned int n = std::thread::hardware_concurrency();
    std::cout << n << " concurrent threads are supported.\n";

    RUN_TEST(Bitmap, TestGray8);
    RUN_TEST(Bitmap, TestGray16);
    RUN_TEST(Bitmap, TestRgb48);
    RUN_TEST(Bitmap, TestRgb24);
    RUN_TEST(Bitmap, TestInterpolation);

    RUN_TEST(PpmDecoder, TestPlain);
    RUN_TEST(PpmDecoder, TestGray8);
    RUN_TEST(PpmDecoder, TestGray16);
    RUN_TEST(PpmDecoder, TestRgb24);
    RUN_TEST(PpmDecoder, TestRgb48);
    RUN_TEST(PpmDecoder, TestByteOrdering);
    RUN_TEST(PpmDecoder, ReadTwice);    
    RUN_TEST(PpmDecoder, ReadStripes);

    RUN_TEST(PpmEncoder, TestGray8);
    RUN_TEST(PpmEncoder, TestGray16);
    RUN_TEST(PpmEncoder, TestRgb24);
    RUN_TEST(PpmEncoder, TestRgb48);

    RUN_TEST(Converter, TestRgb24ToGray8);
    RUN_TEST(Converter, TestRgb48ToGray16);
    RUN_TEST(Converter, TestAstrophoto);

    RUN_TEST(Registrator, BasicTest);
    RUN_TEST(Registrator, TestVertical);
    RUN_TEST(Registrator, RegistrateHugePhoto);
    RUN_TEST(Registrator, TestMultipleTiles);

    RUN_TEST(BinningTransform, TestRGB24);
    RUN_TEST(BinningTransform, TestHugePicture);
    RUN_TEST(BinningTransform, Test3x3);

    RUN_TEST(RawDecoder, TestAttach);
    RUN_TEST(RawDecoder, TestReadBitmap);
    RUN_TEST(RawDecoder, TestDNG);

    RUN_TEST(FastAligner, BasicTest);
    RUN_TEST(FastAligner, InverseMatching);
    RUN_TEST(FastAligner, RealPhotoTest);
    RUN_TEST(FastAligner, TestThreshold60);
    RUN_TEST(FastAligner, TestLargeIntervalPhotos);    

    RUN_TEST(Newton2D, TestAnaliticalJacobian);
    RUN_TEST(Newton2D, TestNumericalJacobian);
    
    RUN_TEST(Stacker, TestEquatorialRegion);
    RUN_TEST(Stacker, TestTwoPics);
    RUN_TEST(Stacker, TestFastStacking);
    RUN_TEST(Stacker, TestThreePics);
    RUN_TEST(Stacker, TestMilkyWay);
    RUN_TEST(Stacker, TestStackingWithoutAlignment);

    return 0;
}
