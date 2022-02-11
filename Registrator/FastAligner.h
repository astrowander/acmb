#ifndef FASTALIGNER_H
#define FASTLIGNER_H

#include <vector>
#include <unordered_map>

#include "star.h"
#include "../AGG/agg_trans_affine.h"

using StarPair = std::pair<Star, Star>;

class FastAligner
{
	std::vector<Star> _refStars;
	std::vector<Star> _targetStars;

	std::unordered_map<size_t, size_t> _matches;

	double _eps = 1.0;
	

	bool TryRefStar(size_t refIndex, std::unordered_map<size_t, size_t>& temp, const agg::trans_affine& transform);

public:

	FastAligner(const std::vector<Star>& refStars);

	void Align(const std::vector<Star>& _targetStars);

	const std::unordered_map<size_t, size_t>& GetMatches();

	void SetEps(double eps);
};
#endif