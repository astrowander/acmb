//#define GENERATE_PATTERNS
#include "testtools.h"
#include "../Core/bitmap.h"
#include "../Tools/SystemTools.h"
#include <cstring>
#include <cstdlib>
#include <filesystem>

ACMB_TESTS_NAMESPACE_BEGIN

bool BitmapsAreEqual(std::shared_ptr<IBitmap> lhs, std::shared_ptr<IBitmap> rhs)
{
    if
    (    lhs->GetPixelFormat() != rhs->GetPixelFormat() ||
            lhs->GetWidth() != rhs->GetWidth() ||
            lhs->GetHeight() != rhs->GetHeight() ||
            lhs->GetByteSize() != rhs->GetByteSize()
    )
        return false;

    return !static_cast<bool>(memcmp(lhs->GetPlanarScanline(0), rhs->GetPlanarScanline(0), lhs->GetByteSize()));
}

bool BitmapsAreEqual(const std::string& fileName, std::shared_ptr<IBitmap> rhs)
{
#ifdef GENERATE_PATTERNS
    const auto dir = fileName.substr(0, fileName.find_last_of("/\\"));
    if ( !std::filesystem::exists(dir) && !std::filesystem::create_directories( dir ))
        throw std::runtime_error(std::string("unable to create directory") + dir);

    IBitmap::Save(rhs, fileName);
    return true;
#else
    return BitmapsAreEqual(IBitmap::Create(fileName, rhs->GetPixelFormat()), rhs);
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
