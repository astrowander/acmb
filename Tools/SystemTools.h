#pragma once

#include "../Core/macros.h"
#include <string>
#include <utility>
#include <stdexcept>

ACMB_NAMESPACE_BEGIN

std::string GetEnv( const std::string& name );
std::string ToLower( const std::string& val );

ACMB_NAMESPACE_END
