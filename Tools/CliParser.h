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

    CliParser( int argc, char** argv );
    std::pair<int, std::string> Parse( bool testMode = false );

public:

    static std::pair<int, std::string> Parse( int argc, char** argv );
    TEST_ACCESS( CliParser );
};
