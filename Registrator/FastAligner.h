#ifndef FASTALIGNER_H
#define FASTALIGNER_H

#include <vector>
#include <unordered_map>

#include "star.h"
#include "../AGG/agg_trans_affine.h"

using StarPair = std::pair<Star, Star>;
using IndexMap = std::unordered_map<size_t, size_t>;
using MatchMap = std::unordered_map<PointF, PointF, PointFHasher>;

class FastAligner
{
	std::vector<Star> _refStars;
	std::vector<Star> _targetStars;

	IndexMap _matches;
	agg::trans_affine _transform;

	double _eps = 1.0;

	std::pair< IndexMap, agg::trans_affine> BruteForceSearch(const size_t n);
	void BruteForceCheckTransform(const size_t refLim, const size_t targetLim, IndexMap& temp, const agg::trans_affine& transform);
	
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

			PointF targetPos = targetStar.center;
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

	FastAligner(const std::vector<Star>& refStars);

	void Align(const std::vector<Star>& targetStars, double eps = 5.0);

	template<class TransformType>
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
	}

	MatchMap GetMatches();
	const agg::trans_affine& GetTransform();

	void SetEps(double eps);
};
#endif