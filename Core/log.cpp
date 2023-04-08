#include "log.h"

ACMB_NAMESPACE_BEGIN

void Log( const std::string& message )
{
    if ( enableLogging )
        std::cout << message << std::endl;
}

ACMB_NAMESPACE_END