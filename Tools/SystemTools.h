#pragma once

#include "../Core/macros.h"
#include <string>
#include <utility>
#include <stdexcept>
#include <random>
#include <mutex>

ACMB_NAMESPACE_BEGIN
/// returns given system environment variable. Throws exception if it doesn't exist
std::string GetEnv( const std::string& name );
/// Converts given string to lower
std::string ToLower( const std::string& val );

class RandomStringGenerator
{
    std::random_device rd;
    std::mt19937 generator;
    std::uniform_int_distribution<> dist;
    std::string charset;
    inline static std::mutex mtx;

public:
    RandomStringGenerator( const std::string& characters = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" );
    std::string operator()( std::size_t length = 10 );   
};

ACMB_NAMESPACE_END
