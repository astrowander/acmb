#define _USE_MATH_DEFINES
#include "FastAligner.h"

using PointFPair = std::pair<PointF, PointF>;

agg::trans_affine CalculateTransform(PointFPair& refPoints, PointFPair& targetPoints)
{
	auto refAngle = atan2(refPoints.second.y - refPoints.first.y, refPoints.second.x - refPoints.first.x);
	auto targetAngle = atan2(targetPoints.second.y - targetPoints.first.y, targetPoints.second.x - targetPoints.first.x);

	auto rotation = refAngle - targetAngle;
	if (rotation > M_PI)
		rotation -= M_PI;
	if (rotation < -M_PI)
		rotation += M_PI;

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

void FastAligner::Align(const std::vector<Star>& _targetStars)
{
	agg::trans_affine transform;
	std::unordered_map<size_t, size_t> temp;
	TryRefStar(0, temp, transform);
}

void FastAligner::TryRefStar(size_t refIndex, std::unordered_map<size_t, size_t>& temp, const agg::trans_affine& transform)
{
	if (refIndex == _refStars.size())
	{
		if (temp.size() > _matches.size())
			_matches = temp;

		return;
	}

	const auto& refStar = _refStars[refIndex];

	for (size_t i = 0; i < _targetStars.size(); ++i)
	{
		auto it = _matches.find(i);
		if (it != std::end(_matches))
			continue;

		temp.insert({ i, refIndex });

		const auto& targetStar = _targetStars[i];
		switch (temp.size())
		{
		case 1:
			TryRefStar(refIndex + 1, temp, transform);
			//translate transform
			break;
		case 2:
		{
			auto it = temp.begin();
			PointFPair refPair{ _refStars[it->second].center, _refStars[refIndex].center };
			PointFPair targetPair{ _targetStars[it->first].center, _targetStars[i].center };

			auto penalty = std::fabs(refPair.first.Distance(refPair.second) - targetPair.first.Distance(targetPair.second));
			if (penalty < 5.0)
			{
				auto newTransform = CalculateTransform(refPair, targetPair);
				TryRefStar(refIndex + 1, temp, newTransform);
			}

			break;
		}
		default:
		{
			PointF targetPos = targetStar.center;
			transform.transform(&targetPos.x, &targetPos.y);
			auto penalty = targetPos.Distance(refStar.center);
			if (penalty < 5.0)
			{
				TryRefStar(refIndex + 1, temp, transform);
			}

			break;
		}
		}

		if (temp.size() > _matches.size())
			_matches = temp;

		temp.erase(i);
	}
}
