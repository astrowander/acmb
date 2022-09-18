#define GENERATE_PATTERNS
#include "testtools.h"
#include "../Core/bitmap.h"
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
    if ( !std::filesystem::exists(dir) && !std::filesystem::create_directory( dir ))
        throw std::runtime_error(std::string("unable to create directory") + dir);

    IBitmap::Save(rhs, fileName);
    return true;
#endif
    return BitmapsAreEqual(IBitmap::Create(fileName), rhs);
}

std::string GetEnv(const std::string& name)
{
    const char* val = std::getenv( name.c_str() );
    if ( !val )
        throw std::runtime_error(std::string("Environment variable ") + name + std::string(" does not exist"));

    return val;
}

const std::string testFilesPath = GetEnv("ACMB_TESTS") + "/TestFiles/";
const std::string patternsPath = GetEnv("ACMB_TESTS") + "/Patterns/";

std::string GetPathToPattern(const std::string &fileName)
{
    return patternsPath + fileName;
}

std::string GetPathToTestFile(const std::string &fileName)
{
    return testFilesPath + fileName;
}

ACMB_TESTS_NAMESPACE_END
