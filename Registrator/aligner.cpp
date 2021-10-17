#include "aligner.h"
#include "registrator.h"

Aligner::Aligner(std::shared_ptr<AlignmentDataset> pRefDataset, std::shared_ptr<AlignmentDataset> pTargetDataset)
: _pRefDataset(pRefDataset)
, _pTargetDataset(pTargetDataset)
{}

void Aligner::Align()
{
    for (uint32_t i = 0; i < _pRefDataset->valuableStarCount - 1; ++i)
    for (uint32_t j = i + 1; j < _pRefDataset->valuableStarCount; ++j)
    {
        if (TryRefPair({ _pRefDataset->stars[i], _pRefDataset->stars[j] }))
            return;
    }

}

bool Aligner::CheckTransform()
{
    uint32_t matches = 0;

    for (uint32_t i = 0; i < _pRefDataset->valuableStarCount; ++i)
    {
        auto refPoint = _pRefDataset->stars[i].center;
        _pTargetDataset->transform.transform(&refPoint.x, &refPoint.y);

        for (uint32_t j = 0; j < _pTargetDataset->valuableStarCount; ++j)
        {
            auto targetPoint = _pTargetDataset->stars[j].center;
            auto dist = refPoint.Distance(targetPoint);
            //BUGBUG magic number
            if (dist < 5)
            {
                ++matches;
                break;
            }
        }
    }

    if (matches > 2)
    {
        std::cout << "match count = " << matches << std::endl;
    }
    return (matches > 2);
}

bool Aligner::TryRefPair(const std::pair<Star, Star>& refPair)
{
    std::pair<PointF, PointF> refPoints{refPair.first.center, refPair.second.center};
    auto refDist = refPoints.first.Distance(refPoints.second);

    std::pair<Star, Star> targetPair { Star{}, Star{} };
    std::pair<PointF, PointF> targetPoints{ PointF{}, PointF{} };
    for (uint32_t i = 0; i < _pTargetDataset->valuableStarCount - 1; ++i)
    {
        targetPair.first = _pTargetDataset->stars[i];
        targetPoints.first = targetPair.first.center;

        for (uint32_t j = i + 1; j < _pTargetDataset->valuableStarCount; ++j)
        {
            targetPair.second = _pTargetDataset->stars[j];
            targetPoints.second = targetPair.second.center;
            auto targetDist = targetPoints.first.Distance(targetPoints.second);

            //BUGBUG magic number
            if (fabs(targetDist - refDist) > 5)
                continue;

            _pTargetDataset->transform = CalculateTransform(refPoints, targetPoints);

            if (CheckTransform())
                return true;

            std::swap(targetPoints.first, targetPoints.second);
            _pTargetDataset->transform = CalculateTransform(refPoints, targetPoints);

            if (CheckTransform())
                return true;
        }
    }

    return false;
}

agg::trans_affine Aligner::CalculateTransform(PointFPair &refPoints, PointFPair &targetPoints)
{
    auto dx = refPoints.first.x - refPoints.second.x;
    auto dy = refPoints.first.y - refPoints.second.y;
    double det = dx * dx + dy * dy;

    auto dx1 = targetPoints.first.x - targetPoints.second.x;
    auto dy1 = targetPoints.first.y - targetPoints.second.y;

    double cosa = (dx1 * dx + dy1 * dy) / det;
    double sina = (dx1 * dy1 - dx1 * dy) / det;

    return agg::trans_affine (cosa, sina, -sina, cosa, targetPoints.first.x + sina * refPoints.first.y - cosa * refPoints.first.x, targetPoints.first.y - cosa * refPoints.first.y - sina * refPoints.first.x);
}

void Aligner::Align(std::shared_ptr<AlignmentDataset> pRefDataset, std::shared_ptr<AlignmentDataset> pTargetDataset)
{
    Aligner aligner(pRefDataset, pTargetDataset);
    return aligner.Align();
}
