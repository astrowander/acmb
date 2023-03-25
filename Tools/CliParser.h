#pragma once

#include "../Core/macros.h"
#include "../Core/pipeline.h"
#include "../Tests/test.h"
#include <string>
#include <memory>

ACMB_TESTS_NAMESPACE_BEGIN
class TestCliParser;
ACMB_TESTS_NAMESPACE_END

ACMB_NAMESPACE_BEGIN

class ImageDecoder;

/// holds key and list of its argumnets
struct KV
{
    std::string key;
    std::vector<std::string> values;
};

/// Parses command line and calls needed functions of the library
class CliParser
{
    std::vector<KV> _kvs;
    std::vector<Pipeline> _pipelinesBeforeStacker;
    std::vector<Pipeline> _darkPipelines;
    Pipeline _pipelineAfterStacker;

    CliParser( int argc, const char** argv );
    std::tuple<int, std::string> Parse( bool testMode = false );

public:

    /// Parse given command line
    static std::tuple<int, std::string> Parse( int argc, const char** argv );
    TEST_ACCESS( CliParser );
};

ACMB_NAMESPACE_END
