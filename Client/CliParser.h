#pragma once
#include "./../Core/macros.h"
#include <string>
#include <vector>

ACMB_CLIENT_NAMESPACE_BEGIN

/// holds key and list of its argumnets
struct KV
{
    std::string key;
    std::vector<std::string> values;
};

std::vector<KV> Parse( const std::vector<std::string>& args );


ACMB_CLIENT_NAMESPACE_END
