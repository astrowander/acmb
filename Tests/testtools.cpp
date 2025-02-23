#define GENERATE_PATTERNS
#include "testtools.h"
#include "../Core/bitmap.h"
#include "../Tools/SystemTools.h"
#include <cstring>
#include <cstdlib>
#include <filesystem>

ACMB_TESTS_NAMESPACE_BEGIN

bool BitmapsAreEqual(std::shared_ptr<IBitmap> expected, std::shared_ptr<IBitmap> actual)
{
    if
    (    expected->GetPixelFormat() != actual->GetPixelFormat() ||
            expected->GetWidth() != actual->GetWidth() ||
            expected->GetHeight() != actual->GetHeight() ||
            expected->GetByteSize() != actual->GetByteSize()
    )
        return false;

    const auto res = memcmp( expected->GetPlanarScanline( 0 ), actual->GetPlanarScanline( 0 ), expected->GetByteSize() );
    return !static_cast<bool>(res);
}

bool BitmapsAreEqual(const std::string& fileName, std::shared_ptr<IBitmap> actual)
{
#ifdef GENERATE_PATTERNS
    const auto dir = fileName.substr(0, fileName.find_last_of("/\\"));
    if ( !std::filesystem::exists(dir) && !std::filesystem::create_directories( dir ))
        throw std::runtime_error(std::string("unable to create directory") + dir);

    IBitmap::Save( actual, fileName);
    return true;
#else
    auto res = BitmapsAreEqual( IBitmap::Create( fileName, actual->GetPixelFormat() ), actual );
    if ( !res )
        IBitmap::Save( actual, fileName + ".actual.ppm");
    return res;
#endif
}

const std::string testFilesPath = GetEnv("ACMB_PATH") + "/Tests/TestFiles/";
const std::string patternsPath = GetEnv("ACMB_PATH") + "/Tests/Patterns/";

std::string GetPathToPattern(const std::string &fileName)
{
    return patternsPath + fileName;
}

std::string GetPathToTestFile(const std::string &fileName)
{
    return testFilesPath + fileName;
}

ACMB_TESTS_NAMESPACE_END
