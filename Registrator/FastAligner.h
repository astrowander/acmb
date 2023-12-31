#pragma once

#include "star.h"

#include "../AGG/agg_trans_affine.h"
#include "../Libs/parallel-hashmap/parallel_hashmap/phmap.h"

#include <vector>
#include <array>

using phmap::parallel_flat_hash_map;

ACMB_NAMESPACE_BEGIN

constexpr uint32_t bruteForceSearchSize = 30;

using StarPair = std::pair<Star, Star>;
//using IndexMap = parallel_flat_hash_map<size_t, size_t>;
//using IndexMap = std::vector<int>;
class IndexMap
{
    std::vector<int> _map;
    size_t _size;

public:
    IndexMap( size_t capacity = 0 ) 
	: _map( capacity, -1 )
	, _size( 0 ) 
	{}

    int operator[]( size_t i ) const 
	{
        return _map[i];
    }

    void set( size_t i, int value )
    {
        if ( _map[i] == -1 && value != -1 )
            ++_size;
        else if ( _map[i] != -1 && value == -1 )
            --_size;

        _map[i] = value;
    }
    size_t size() const 
	{
        return _size;
    }
};
using MatchMap = parallel_flat_hash_map<PointF, PointF, PointFHasher>;
using BruteForceIndexMap = std::pair< std::array<uint8_t, bruteForceSearchSize>, uint8_t>;
/// <summary>
/// This class receives two vectors of stars and finds respective stars in them.
/// </summary>
class FastAligner
{
	std::vector<Star> _refStars;
	std::vector<Star> _targetStars;

	IndexMap _matches;
	agg::trans_affine _transform;

	double _eps = 1.0;

	agg::trans_affine BruteForceSearch(const size_t n);
	BruteForceIndexMap BruteForceCheckTransform(const size_t refLim, const size_t targetLim, const std::pair<uint8_t, uint8_t>& refs, const std::pair<uint8_t, uint8_t>& targets, const agg::trans_affine& transform);
	
	template <class TransformType>
	bool TryRefStar(size_t refIndex, IndexMap& matches, const TransformType& transform)
	{
		if (refIndex == _refStars.size())
		{
			if (matches.size() > _matches.size() && matches.size() > 2)
			{
				_matches = matches;
				return true;
			}

			return false;
		}

		const auto& refStar = _refStars[refIndex];

		for (size_t i = 0; i < _targetStars.size(); ++i)
		{
			if ( matches[i] != -1 )
                continue;

			matches.set( i, refIndex );
			const auto& targetStar = _targetStars[i];

			PointF targetPos = targetStar.center;
			transform.transform(&targetPos.x, &targetPos.y);
			auto penalty = targetPos.Distance(refStar.center);
			if (penalty < _eps)
			{
				if (TryRefStar(refIndex + 1, matches, transform))
					return true;
			}

            matches.set( i, - 1);
		}

		if (TryRefStar(refIndex + 1, matches, transform))
			return true;

		return false;
	}

public:
	/// Creates object with reference vector of stars
	FastAligner(const std::vector<Star>& refStars);
	/// Receives the target vector of stars and finds respective stars to the reference vector
	void Align(const std::vector<Star>& targetStars, double eps = 5.0);

	/// Receives the target vector of stars, applies given transform and finds respective stars to the reference vector
	/*template<class TransformType>
	void Align(const std::vector<Star>& targetStars, const TransformType& transform, double eps = 5.0)
	{
		_eps = eps;
		_targetStars = targetStars;
		_matches.clear();

		for (size_t i = 0; i < _refStars.size(); ++i)
		{
			MatchMap temp;
			if (TryRefStar(i, temp, transform))
				return;
		}
	}*/
	/// Returns map of respective stars
	MatchMap GetMatches() const;
	/// Returns found transform
	const agg::trans_affine& GetTransform() const;
	/// Sets maximal error
	void SetEps(double eps);
};

ACMB_NAMESPACE_END
