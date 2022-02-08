#ifndef ALIGNER_H
#define ALIGNER_H

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
	

	void TryRefStar(size_t refIndex, std::unordered_map<size_t, size_t>& temp, const agg::trans_affine& transform);

public:

	FastAligner(const std::vector<Star>& refStars);

	void Align(const std::vector<Star>& _targetStars);
};
#endif