#define _USE_MATH_DEFINES
#include "FastAligner.h"

ACMB_NAMESPACE_BEGIN

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



void FastAligner::Align(const std::vector<Star>& targetStars, double eps)
{
	_eps = eps;
	_targetStars = targetStars;
    _matches = IndexMap( _targetStars.size() );
	_transform = BruteForceSearch(bruteForceSearchSize);
	auto matches = _matches;
	for (size_t i = bruteForceSearchSize + 1; i < _refStars.size(); ++i)
	{
		IndexMap temp( matches );
		if (TryRefStar(i, temp, _transform))
			return;
	}
}

MatchMap FastAligner::GetMatches() const
{
	MatchMap res;
	for (size_t i = 0; i < _matches.size(); ++i)
	{
        if ( _matches[i] == -1 )
            continue;
		res.insert({ _targetStars[i].center, _refStars[_matches[i]].center });
	}

	return res;
}

const agg::trans_affine& FastAligner::GetTransform() const
{
	return _transform;
}

void FastAligner::SetEps(double eps)
{
	_eps = eps;
}

agg::trans_affine  FastAligner::BruteForceSearch(const size_t N)
{
	agg::trans_affine res;
	if ( _refStars.size() == 0 || _targetStars.size() == 0 )
		return res;

	const auto refLim = std::min<size_t>(_refStars.size(), N);
	const auto targetLim = std::min<size_t>(_targetStars.size(), N);

	BruteForceIndexMap resMatches{ {}, 2 };

	for ( uint8_t i = 0; i < refLim - 1; ++i)
	for ( uint8_t j = i + 1; j < refLim; ++j)
	{
		PointFPair refPair{ _refStars[i].center, _refStars[j].center };

		for ( uint8_t k = 0; k < targetLim - 1; ++k)
		for ( uint8_t l = k + 1; l < targetLim; ++l)
		{
			PointFPair targetPair{ _targetStars[k].center, _targetStars[l].center };
			auto penalty = std::fabs(refPair.first.Distance(refPair.second) - targetPair.first.Distance(targetPair.second));
			if (penalty > _eps)
				continue;

			auto transform = CalculateTransform(refPair, targetPair);
			auto matches = BruteForceCheckTransform(refLim, targetLim, {i,j}, {k,l}, transform);
			if ( matches.second > resMatches.second )
			{
				resMatches = std::move( matches );
				res = transform;
			}

			//IndexMap temp2 = IndexMap{ {k, j}, {l, i} };
			transform = CalculateTransform(refPair, { _targetStars[l].center , _targetStars[k].center });
			matches = BruteForceCheckTransform( refLim, targetLim, { j,i }, { k,l }, transform );
			if ( matches.second > resMatches.second )
			{
				resMatches = std::move( matches );
				res = transform;
			}
		}
	}

	for ( uint8_t i = 0; i < bruteForceSearchSize; ++i )
	{
		if ( resMatches.first[i] != 0 )
			_matches.set( i, resMatches.first[i] - 1 );
	}

	return res;
}

BruteForceIndexMap FastAligner::BruteForceCheckTransform( const size_t refLim, const size_t targetLim, const std::pair<uint8_t, uint8_t>& refs, const std::pair<uint8_t, uint8_t>& targets, const agg::trans_affine& transform )
{
	BruteForceIndexMap res;
	res.first[targets.first] = refs.first + 1;
	res.first[targets.second] = refs.second + 1;
	res.second = 2;

	const auto squaredEps = _eps * _eps;

	for (uint8_t i = 0; i < targetLim; ++i)
	{
		if (targets.first == i || targets.second == i)
			continue;

		auto transformedRefPoint = _targetStars[i].center;
		transform.transform(&transformedRefPoint.x, &transformedRefPoint.y);

		for ( uint8_t j = 0; j < refLim; ++j)
		{
			if (refs.first == j || refs.second == j)
				continue;

			if ( transformedRefPoint.SquaredDistance(_refStars[j].center) > squaredEps )
				continue;

			res.first[i] = j + 1;
			++res.second;
			break;
		}
	}

	return res;
}

ACMB_NAMESPACE_END