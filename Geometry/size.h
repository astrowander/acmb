#pragma once
#include <cstdint>
#include <iostream>
ACMB_NAMESPACE_BEGIN
/// <summary>
/// Size of an image, or of a region in it
/// </summary>
template <typename T>
struct SizeT
{
	T width;
	T height;

	bool operator==(const SizeT& rhs) const
	{
		return width == rhs.width && height == rhs.height;
	}
	/// prints size to a stream
	template <typename U>
	friend std::ostream& operator <<(std::ostream& out, const SizeT<U>& size);
};

template <typename T>
std::ostream& operator<<(std::ostream& out, const SizeT<T>& size)
{
    return out << size.width << " " << size.height;
}
/// size with integer params
using Size = SizeT<uint32_t>;
/// size with fractional params
using SizeF = SizeT<double>;

ACMB_NAMESPACE_END