#include "CliParser.h"
#include <stdexcept>

ACMB_CLIENT_NAMESPACE_BEGIN

std::vector<KV> Parse( const std::vector<std::string>& args )
{
    std::vector<KV> res;

    for ( const auto& arg: args )
    {
        if ( arg[0] == '-' && arg[1] == '-' )
        {
            res.push_back( { arg, {} } );
            continue;
        }

        if ( res.empty() )
            throw std::invalid_argument( "no key" );

        auto& kv = res.back();
        kv.values.push_back( arg );
    }

    return res;
}

ACMB_CLIENT_NAMESPACE_END
