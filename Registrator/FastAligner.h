#ifndef FASTALIGNER_H
#define FASTLIGNER_H

#include <vector>
#include <unordered_map>

#include "star.h"
#include "../AGG/agg_trans_affine.h"

using StarPair = std::pair<Star, Star>;
using MatchMap = std::unordered_map<size_t, size_t>;
class FastAligner
{
	std::vector<Star> _refStars;
	std::vector<Star> _targetStars;

	MatchMap _matches;
	agg::trans_affine _transform;

	double _eps = 1.0;

	std::pair< std::unordered_map<size_t, size_t>, agg::trans_affine> BruteForceSearch(const size_t n);
	void BruteForceCheckTransform(const size_t refLim, const size_t targetLim, MatchMap& temp, const agg::trans_affine& transform);
	

	bool TryRefStar(size_t refIndex, MatchMap& matches, const agg::trans_affine& transform, bool needToCalculateTransform);

public:

	FastAligner(const std::vector<Star>& refStars);

	void Align(const std::vector<Star>& targetStars, double eps = 5.0);
	void Align(const std::vector<Star>& targetStars, const agg::trans_affine& transform, double eps = 5.0);

	const MatchMap& GetMatches();
	const agg::trans_affine& GetTransform();

	void SetEps(double eps);
};
#endif