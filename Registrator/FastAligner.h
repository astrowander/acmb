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
using IndexMap = parallel_flat_hash_map<size_t, size_t>;
using MatchMap = parallel_flat_hash_map<PointD, PointD, PointDHasher>;
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

	std::pair< IndexMap, agg::trans_affine> BruteForceSearch(const size_t n);
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
			auto it = matches.find(i);
			if (it != std::end(matches))
				continue;

			matches.insert({ i, refIndex });

			const auto& targetStar = _targetStars[i];

			PointD targetPos = targetStar.center;
			transform.transform(&targetPos.x, &targetPos.y);
			auto penalty = targetPos.Distance(refStar.center);
			if (penalty < _eps)
			{
				if (TryRefStar(refIndex + 1, matches, transform))
					return true;
			}

			matches.erase(i);
		}

		if (TryRefStar(refIndex + 1, matches, transform))
			return true;

		return false;
	}

public:
	/// Creates object with reference vector of stars
	FastAligner(const std::vector<Star>& refStars);
	/// Receives the target vector of stars and finds respective stars to the reference vector
	void Align(const std::vector<Star>& targetStars, bool simpleMode = false, double eps = 5.0);

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
