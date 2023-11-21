#include "SystemTools.h"
#include <filesystem>

ACMB_NAMESPACE_BEGIN

std::string GetEnv( const std::string& name )
{
#ifdef _WIN32
	char* val = nullptr;
	size_t count = 0;
	if ( _dupenv_s( &val, &count, name.c_str() ) || !val )
		throw std::runtime_error( std::string( "Environment variable " ) + name + std::string( " does not exist" ) );

	return val;
#else
	const char* val = std::getenv( name.c_str() );
	if ( !val )
		throw std::runtime_error( std::string( "Environment variable " ) + name + std::string( " does not exist" ) );

	return val;
#endif
}

std::string GetAcmbPath()
{
	try
	{
		return GetEnv( "ACMB_PATH" );
	}
	catch ( std::exception& )
	{
		return std::filesystem::current_path().string();
	}
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

RandomStringGenerator::RandomStringGenerator( const std::string& characters )
: generator( rd() ), dist( 0, int( characters.size() ) - 1), charset(characters)
{
}

std::string RandomStringGenerator::operator()( size_t length )
{
	std::lock_guard<std::mutex> lock( mtx );  // Make sure this function is thread safe.

	std::string result( length, '\0' );
	for ( std::size_t i = 0; i < length; ++i )
	{
		result[i] = charset[dist( generator )];
	}
	return result;
}

ACMB_NAMESPACE_END
