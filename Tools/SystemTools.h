#pragma once

#include "../Core/macros.h"
#include <string>
#include <utility>
#include <stdexcept>

ACMB_NAMESPACE_BEGIN
/// returns given system environment variable. Throws exception if it doesn't exist
std::string GetEnv( const std::string& name );
/// Converts given string to lower
std::string ToLower( const std::string& val );

ACMB_NAMESPACE_END
