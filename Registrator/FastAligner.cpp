#define _USE_MATH_DEFINES
#include "FastAligner.h"

using PointFPair = std::pair<PointF, PointF>;

agg::trans_affine CalculateTransform(const PointFPair& refPoints, const PointFPair& targetPoints)
{
	auto refAngle = atan2(refPoints.second.y - refPoints.first.y, refPoints.second.x - refPoints.first.x);
	if (refAngle < 0)
		refAngle += 2 * M_PI;

	auto targetAngle = atan2(targetPoints.second.y - targetPoints.first.y, targetPoints.second.x - targetPoints.first.x);
	if (targetAngle < 0)
		targetAngle += 2 * M_PI;

	auto rotation = refAngle - targetAngle;
	auto rotateMatrix = agg::trans_affine_rotation(rotation);

	auto targetPoint = targetPoints.first;
	rotateMatrix.transform(&targetPoint.x, &targetPoint.y);

	auto dx = refPoints.first.x - targetPoint.x;
	auto dy = refPoints.first.y - targetPoint.y;
	auto translate_matrix = agg::trans_affine_translation(dx, dy);

	auto res = rotateMatrix * translate_matrix;
	return res;
}

FastAligner::FastAligner(const std::vector<Star>& refStars)
	: _refStars(refStars)
{

}

constexpr uint32_t bruteForceSearchSize = 30;

void FastAligner::Align(const std::vector<Star>& targetStars, double eps)
{
	_eps = eps;
	_targetStars = targetStars;
	_matches.clear();	

	auto res = BruteForceSearch(bruteForceSearchSize);
	_matches = res.first;
	_transform = res.second;

	for (size_t i = bruteForceSearchSize + 1; i < _refStars.size(); ++i)
	{
		IndexMap temp(res.first);
		if (TryRefStar(i, temp, res.second))
			return;
	}
}

MatchMap FastAligner::GetMatches()
{
	MatchMap res;
	for (auto it : _matches)
	{
		res.insert({ _targetStars[it.first].center, _refStars[it.second].center });
	}

	return res;
}

const agg::trans_affine& FastAligner::GetTransform()
{
	return _transform;
}

void FastAligner::SetEps(double eps)
{
	_eps = eps;
}

std::pair< IndexMap, agg::trans_affine>  FastAligner::BruteForceSearch(const size_t N)
{
	std::pair< IndexMap, agg::trans_affine> res;
	if ( _refStars.size() == 0 || _targetStars.size() == 0 )
		return res;

	const auto refLim = std::min<size_t>(_refStars.size(), N);
	const auto targetLim = std::min<size_t>(_targetStars.size(), N);

	for (size_t i = 0; i < refLim - 1; ++i)
	for (size_t j = i + 1; j < refLim; ++j)
	{
		PointFPair refPair{ _refStars[i].center, _refStars[j].center };

		for (size_t k = 0; k < targetLim - 1; ++k)
		for (size_t l = k + 1; l < targetLim; ++l)
		{
			PointFPair targetPair{ _targetStars[k].center, _targetStars[l].center };
			auto penalty = std::fabs(refPair.first.Distance(refPair.second) - targetPair.first.Distance(targetPair.second));
			if (penalty > _eps)
				continue;

			IndexMap temp {{k, i}, {l, j}};			
			auto transform = CalculateTransform(refPair, targetPair);
			BruteForceCheckTransform(refLim, targetLim, temp, transform);
			if (temp.size() > res.first.size())
			{
				res.first = temp;
				res.second = transform;
			}

			temp = IndexMap{ {k, j}, {l, i} };
			transform = CalculateTransform(refPair, { _targetStars[l].center , _targetStars[k].center });
			BruteForceCheckTransform(refLim, targetLim, temp, transform);
			if (temp.size() > res.first.size())
			{
				res.first = temp;
				res.second = transform;
			}
		}
	}

	return res;
}

void FastAligner::BruteForceCheckTransform(const size_t refLim, const size_t targetLim, IndexMap& temp, const agg::trans_affine& transform)
{
	size_t refs[2] = { temp.begin()->second, std::next(temp.begin())->second };
	size_t targets[2] = { temp.begin()->first, std::next(temp.begin())->first };

	for (size_t i = 0; i < targetLim; ++i)
	{
		if (targets[0] == i || targets[1] == i)
			continue;

		auto transformedRefPoint = _targetStars[i].center;
		transform.transform(&transformedRefPoint.x, &transformedRefPoint.y);

		for (size_t j = 0; j < refLim; ++j)
		{
			if (refs[0] == j || refs[1] == j)
				continue;

			if (transformedRefPoint.Distance(_refStars[j].center) > _eps)
				continue;

			temp[i] = j;
			break;
		}

	}
}
