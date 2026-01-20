#pragma once
#include "./../Core/macros.h"

#include <istream>
#include <ostream>
#include <vector>
#include <string>

ACMB_GUI_NAMESPACE_BEGIN

template<typename T>
struct is_vector : std::false_type
{
};

template<typename T, typename Alloc>
struct is_vector<std::vector<T, Alloc>> : std::true_type
{
};

template<typename T>
inline constexpr bool is_vector_v = is_vector<T>::value;


template<typename T>
struct is_map : std::false_type
{
};

template<typename K, typename V, typename Cmp, typename Alloc>
struct is_map<std::map<K, V, Cmp, Alloc>> : std::true_type
{
};

template<typename T>
inline constexpr bool is_map_v = is_map<T>::value;

template<typename T>
int GetSerializedStringSize( const T& val )
{
    using U = std::remove_cvref_t<T>;

    if constexpr ( std::is_same_v<U, std::string> )
    {
        return sizeof(int) + int(val.size());
    }
    else if constexpr ( is_vector_v<U> )
    {
        int res = sizeof(int);
        for ( const auto& e : val )
            res += GetSerializedStringSize(e);
        return res;
    }
    else if constexpr ( is_map_v<U> )
    {
        int res = sizeof(int);
        for ( const auto& [k, v] : val )
        {
            res += GetSerializedStringSize(k);
            res += GetSerializedStringSize(v);
        }
        return res;
    }
    else
    {
        return sizeof(U);
    }
}

template<typename T>
void Serialize( T&& val, std::ostream& out )
{
    using U = std::remove_cvref_t<T>;

    if constexpr ( std::is_same_v<U, std::string> )
    {
        Serialize(int(val.size()), out);
        out.write(val.data(), val.size());
    }
    else if constexpr ( is_vector_v<U> )
    {
        Serialize(int(val.size()), out);
        for ( auto& e : val )
            Serialize(e, out);
    }
    else if constexpr ( is_map_v<U> )
    {
        Serialize(int(val.size()), out);
        for ( auto& [k, v] : val )
        {
            Serialize(k, out);
            Serialize(v, out);
        }
    }
    else
    {
        out.write(reinterpret_cast<const char*>(&val), sizeof(U));
    }
}

template<typename T>
T Deserialize( std::istream& in, int& remainingBytes )
{
    using U = std::remove_cvref_t<T>;

    auto require = [&](int bytes)
    {
        if ( remainingBytes < bytes )
        {
            in.seekg(remainingBytes, std::ios_base::cur);
            remainingBytes = 0;
            return false;
        }
        return true;
    };

    if constexpr ( std::is_same_v<U, std::string> )
    {
        if ( !require(sizeof(int)) )
            return {};

        int size = Deserialize<int>(in, remainingBytes);
        if ( size <= 0 || !require(size) )
            return {};

        std::string s(size, '\0');
        in.read(s.data(), size);
        remainingBytes -= size;
        return s;
    }
    else if constexpr ( is_vector_v<U> )
    {
        if ( !require(sizeof(int)) )
            return {};

        int size = Deserialize<int>(in, remainingBytes);
        if ( size <= 0 )
            return {};

        U vec;
        vec.reserve(size);
        for ( int i = 0; i < size; ++i )
            vec.push_back(Deserialize<typename U::value_type>(in, remainingBytes));

        return vec;
    }
    else if constexpr ( is_map_v<U> )
    {
        if ( !require(sizeof(int)) )
            return {};

        int size = Deserialize<int>(in, remainingBytes);
        if ( size <= 0 )
            return {};

        U map;
        for ( int i = 0; i < size; ++i )
        {
            auto key = Deserialize<typename U::key_type>(in, remainingBytes);
            auto val = Deserialize<typename U::mapped_type>(in, remainingBytes);
            map.insert_or_assign(std::move(key), std::move(val));
        }
        return map;
    }
    else
    {
        if ( !require(sizeof(U)) )
            return {};

        U res;
        in.read(reinterpret_cast<char*>(&res), sizeof(U));
        remainingBytes -= sizeof(U);
        return res;
    }
}

ACMB_GUI_NAMESPACE_END
