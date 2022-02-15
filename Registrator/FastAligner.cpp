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

constexpr uint32_t bruteForceSearchSize = 40;

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
		MatchMap temp(res.first);
		if (TryRefStar(i, temp, res.second, true))
			return;
	}
}

void FastAligner::Align(const std::vector<Star>& targetStars, const agg::trans_affine& transform, double eps)
{
	_eps = eps;
	_targetStars = targetStars;
	_matches.clear();

	for (size_t i = 0; i < _refStars.size(); ++i)
	{
		MatchMap temp;
		if (TryRefStar(i, temp, transform, false))
			return;
	}
}

const MatchMap& FastAligner::GetMatches()
{
	return _matches;
}

const agg::trans_affine& FastAligner::GetTransform()
{
	return _transform;
}

void FastAligner::SetEps(double eps)
{
	_eps = eps;
}

std::pair< MatchMap, agg::trans_affine>  FastAligner::BruteForceSearch(const size_t N)
{
	std::pair< MatchMap, agg::trans_affine> res;

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

			MatchMap temp {{k, i}, {l, j}};			
			auto transform = CalculateTransform(refPair, targetPair);
			BruteForceCheckTransform(refLim, targetLim, temp, transform);
			if (temp.size() > res.first.size())
			{
				res.first = temp;
				res.second = transform;
			}

			temp = MatchMap{ {k, j}, {l, i} };
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

void FastAligner::BruteForceCheckTransform(const size_t refLim, const size_t targetLim, MatchMap& temp, const agg::trans_affine& transform)
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

bool FastAligner::TryRefStar(size_t refIndex, MatchMap& temp, const agg::trans_affine& transform, bool needToCalculateTransform)
{
	if (refIndex == _refStars.size())
	{
		if (temp.size() > _matches.size() && temp.size() > 2)
		{
			_matches = temp;
			return true;
		}

		return false;
	}

	const auto& refStar = _refStars[refIndex];

	for (size_t i = 0; i < _targetStars.size(); ++i)
	{
		auto it = temp.find(i);
		if (it != std::end(temp))
			continue;

		temp.insert({ i, refIndex });

		const auto& targetStar = _targetStars[i];
		if (needToCalculateTransform && temp.size() == 1)
		{
			if (TryRefStar(refIndex + 1, temp, transform, needToCalculateTransform))
				return true;
		}
		else if (needToCalculateTransform && temp.size() == 2)
		{
			auto it = temp.begin();
			if (it->second == refIndex)
				it = std::next(it);

			PointFPair refPair{ _refStars[it->second].center, _refStars[refIndex].center };
			PointFPair targetPair{ _targetStars[it->first].center, _targetStars[i].center };

			auto penalty = std::fabs(refPair.first.Distance(refPair.second) - targetPair.first.Distance(targetPair.second));

			if (penalty < _eps)
			{
				auto newTransform = CalculateTransform(refPair, targetPair);
				if (TryRefStar(refIndex + 1, temp, newTransform, needToCalculateTransform))
					return true;
			}
		}
		else
		{
			PointF targetPos = targetStar.center;
			transform.transform(&targetPos.x, &targetPos.y);
			auto penalty = targetPos.Distance(refStar.center);
			if (penalty < _eps)
			{
				if (TryRefStar(refIndex + 1, temp, transform, needToCalculateTransform))
					return true;
			}
		}		

		temp.erase(i);
	}

	if (TryRefStar(refIndex + 1, temp, transform, needToCalculateTransform))
		return true;

	return false;
}
