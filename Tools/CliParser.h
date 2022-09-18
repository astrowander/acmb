#pragma once

#include "../Core/macros.h"
#include "../Tests/test.h"
#include <string>
#include <unordered_map>
#include <memory>

ACMB_TESTS_NAMESPACE_BEGIN
class TestCliParser;
ACMB_TESTS_NAMESPACE_END

ACMB_NAMESPACE_BEGIN

class ImageDecoder;

class CliParser
{
    std::unordered_map<std::string, std::string> _kv;
    std::vector<std::shared_ptr<ImageDecoder>> _decoders;
    std::vector<std::shared_ptr<ImageDecoder>> _darks;

    CliParser( int argc, const char** argv );
    std::tuple<int, std::string> Parse( bool testMode = false );

public:

    static std::tuple<int, std::string> Parse( int argc, const char** argv );
    TEST_ACCESS( CliParser );
};

ACMB_NAMESPACE_END
