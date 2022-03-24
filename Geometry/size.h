#pragma once
#include <cstdint>
#include <iostream>

template <typename T>
struct SizeT
{
	T width;
	T height;

	bool operator==(const SizeT& rhs) const
	{
		return width == rhs.width && height == rhs.height;
	}

	template <typename U>
	friend std::ostream& operator <<(std::ostream& out, const SizeT<U>& size);
};

template <typename T>
std::ostream& operator<<(std::ostream& out, const SizeT<T>& size)
{
	out << size.width << " " << size.height;
}

using Size = SizeT<uint32_t>;
using SizeF = SizeT<double>;