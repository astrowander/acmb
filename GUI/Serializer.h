#pragma once
#include "./../Core/macros.h"
#include "./../Transforms/BitmapHealer.h"

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

    if constexpr ( std::is_same_v<std::remove_cvref_t<T>, std::vector<BitmapHealer::Patch>> )
    {
        int res = sizeof( int );
        for ( const auto& patch : val )
            res += GetSerializedStringSize( patch );
        return res;
    }

    if constexpr ( std::is_same_v<std::remove_cvref_t<T>, std::map<int, int>> )
    {
        return int( sizeof( int ) * (2 * val.size() + 1) );
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

    if constexpr ( std::is_same_v<std::remove_cvref_t<T>, std::vector<BitmapHealer::Patch>> )
    {
        Serialize( int( val.size() ), out );
        for ( auto& patch : val )
            Serialize( std::move( patch ), out );
        return;
    }

    if constexpr ( std::is_same_v<std::remove_cvref_t<T>, std::map<int, int>> )
    {
        Serialize( int( val.size() ), out );
        for ( auto it : val )
        {
            Serialize( it.first, out );
            Serialize( it.second, out );
        }
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
        if ( size <= 0 )
            remainingBytes = 0;

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
        if ( size <= 0 )
        {
            remainingBytes = 0;
            return {};
        }
        std::vector<std::string> vec( size );
        for ( int i = 0; i < size; ++i )
            vec[i] = Deserialize<std::string>( in, remainingBytes );

        return vec;
    }

    if constexpr ( std::is_same_v<std::remove_cvref_t<T>, std::vector<BitmapHealer::Patch>> )
    {
        if ( remainingBytes < int( sizeof( int ) ) )
        {
            in.seekg( remainingBytes, std::ios_base::cur );
            remainingBytes = 0;
            return {};
        }

        int size = Deserialize<int>( in, remainingBytes );
        if ( size <= 0 )
        {
            remainingBytes = 0;
            return {};
        }

        std::vector<BitmapHealer::Patch> vec( size );
        for ( int i = 0; i < size; ++i )
            vec[i] = Deserialize<BitmapHealer::Patch>( in, remainingBytes );

        return vec;
    }

    if constexpr ( std::is_same_v<std::remove_cvref_t<T>, std::map<int, int>> )
    {
        if ( remainingBytes < int( sizeof( int ) ) )
        {
            in.seekg( remainingBytes, std::ios_base::cur );
            remainingBytes = 0;
            return {};
        }

        int size = Deserialize<int>( in, remainingBytes );
        if ( size <= 0 )
        {
            remainingBytes = 0;
            return {};
        }
        std::map<int, int> map;
        for ( int i = 0; i < size; ++i )
        {
            int key = Deserialize<int>( in, remainingBytes );
            int value = Deserialize<int>( in, remainingBytes );
            map.insert_or_assign( key, value );
        }
        return map;
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
