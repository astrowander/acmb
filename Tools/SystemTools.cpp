#include "SystemTools.h"

ACMB_NAMESPACE_BEGIN

std::string GetEnv( const std::string& name )
{
#ifdef _WIN32
	char* val = nullptr;
	size_t count = 0;
	if ( _dupenv_s( &val, &count, name.c_str() ) )
		throw std::runtime_error( std::string( "Environment variable " ) + name + std::string( " does not exist" ) );

	return val;
#else
	const char* val = std::getenv( name.c_str() );
	if ( !val )
		throw std::runtime_error( std::string( "Environment variable " ) + name + std::string( " does not exist" ) );

	return val;
#endif
}

std::string ToLower( const std::string& val )
{
	std::string res;
	res.reserve( val.size() );
	for ( auto& ch : val )
	{
		res.push_back( std::tolower( ch ) );
	}
	return res;
}

ACMB_NAMESPACE_END