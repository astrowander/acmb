#pragma once
#include "./../Core/macros.h"
#include <istream>
#include <ostream>
#include <vector>
#include <string>

ACMB_GUI_NAMESPACE_BEGIN

template<typename T>
void Serialize(T&& val, std::ostream& out)
{
    if constexpr (std::is_same_v<std::remove_reference_t<T>, std::string>)
    {
        Serialize( int( val.size() ), out);
        out.write(val.data(), val.size());
        return;
    }

    if constexpr (std::is_same_v<std::remove_reference_t<T>, std::vector<std::string>>)
    {
        Serialize(int( val.size() ), out);
        for (auto& str : val)
            Serialize(str);
        return;
    }

    out.write((char*)(&val), sizeof(T));   
}

template<typename T>
T Deserialize(std::istream& in)
{
    if constexpr (std::is_same_v<T, std::string>)
    {
        size_t size = size_t( Deserialize<int>( in ) );
        std::string str(size, '\0');
        in.read(&str[0], size);
        return str;
    }

    if constexpr (std::is_same_v<T, std::vector<std::string>>)
    {
        size_t size = size_t(Deserialize<int>(in));
        std::vector<std::string> vec( size );
        for (size_t i = 0; i < size; ++i)
            vec[i] = Deserialize<std::string>(in);

        return vec;
    }

    T res;
    in.read((char*)(&res), sizeof(T));
    return res;
}

ACMB_GUI_NAMESPACE_END
