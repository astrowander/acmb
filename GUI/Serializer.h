#pragma once
#include "./../Core/macros.h"
#include <istream>
#include <ostream>
#include <vector>
#include <string>

ACMB_GUI_NAMESPACE_BEGIN

template<typename T>
int GetSerializedStringSize( const T& val )
{
    if constexpr ( std::is_same_v<std::remove_cvref_t<T>, std::string> )
        return int( val.size() ) + sizeof( int );

    if constexpr ( std::is_same_v<std::remove_cvref_t<T>, std::vector<std::string>> )
    {
        int res = sizeof( int );
        for ( const auto& str : val )
            res += GetSerializedStringSize( str );
        return res;
    }

    return sizeof( T );
}

template<typename T>
void Serialize( T&& val, std::ostream& out )
{
    if constexpr ( std::is_same_v<std::remove_cvref_t<T>, std::string> )
    {
        Serialize( int( val.size() ), out );
        out.write( val.data(), val.size() );
        return;
    }

    if constexpr ( std::is_same_v<std::remove_cvref_t<T>, std::vector<std::string>> )
    {
        Serialize( int( val.size() ), out );
        for ( auto& str : val )
            Serialize( std::move( str ), out );
        return;
    }

    out.write( ( char* ) (&val), sizeof( T ) );
}

template<typename T>
T Deserialize( std::istream& in, int& remainingBytes )
{
    if constexpr ( std::is_same_v<std::remove_cvref_t<T>, std::string> )
    {
        if ( remainingBytes < int( sizeof( int ) ) )
        {
            in.seekg( remainingBytes, std::ios_base::cur );
            remainingBytes = 0;
            return {};
        }

        int cachedRemainingBytes = remainingBytes;
        int size = std::min( Deserialize<int>( in, remainingBytes ), cachedRemainingBytes );
        if ( remainingBytes == 0 )
            return {};

        std::string str( size, '\0' );
        in.read( &str[0], size );
        remainingBytes -= size;
        return str;
    }

    if constexpr ( std::is_same_v<std::remove_cvref_t<T>, std::vector<std::string>> )
    {
        if ( remainingBytes < int( sizeof( int ) ) )
        {
            in.seekg( remainingBytes, std::ios_base::cur );
            remainingBytes = 0;
            return {};
        }

        int size = Deserialize<int>( in, remainingBytes );
        std::vector<std::string> vec( size );
        for ( int i = 0; i < size; ++i )
            vec[i] = Deserialize<std::string>( in, remainingBytes );

        return vec;
    }

    if ( remainingBytes < int( sizeof( T )  ) )
    {
        in.seekg( remainingBytes, std::ios_base::cur );
        remainingBytes = 0;
        return {};
    }

    T res;
    in.read( ( char* ) (&res), sizeof( T ) );
    remainingBytes -= sizeof( T );
    return res;
}

ACMB_GUI_NAMESPACE_END
