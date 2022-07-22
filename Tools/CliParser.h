#pragma once
#include <string>
#include <unordered_map>
#include <memory>
#include "../Tests/test.h"

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
