#define _USE_MATH_DEFINES
#include "aligner.h"
#include "registrator.h"

Aligner::Aligner(std::shared_ptr<AlignmentDataset> pRefDataset, std::shared_ptr<AlignmentDataset> pTargetDataset)
: _pRefDataset(pRefDataset)
, _pTargetDataset(pTargetDataset)
{}

void Aligner::Align(std::shared_ptr<AlignmentDataset> pTargetDataset)
{
    _pTargetDataset = pTargetDataset;

    std::vector<double> txs;
    std::vector<double> tys;
    std::vector<double> rotations;

    for (uint32_t i = 0; i < _pRefDataset->stars.size() - 1; ++i)
    for (uint32_t j = i + 1; j < _pRefDataset->stars.size(); ++j)
    {
        if (TryRefPair({ _pRefDataset->stars[i], _pRefDataset->stars[j] }))
        {
            txs.push_back(_pTargetDataset->transform.tx);
            tys.push_back(_pTargetDataset->transform.ty);
            auto rotation = _pTargetDataset->transform.rotation();
            if (rotation == rotation)
                rotations.push_back(rotation);
        }
    }

    auto medianTx = std::begin(txs) + txs.size() / 2;
    std::nth_element(std::begin(txs), medianTx, std::end(txs));

    auto medianTy = std::begin(tys) + tys.size() / 2;
    std::nth_element(std::begin(tys), medianTy, std::end(tys));

    auto medianRotation = std::begin(rotations) + rotations.size() / 2;
    std::nth_element(std::begin(rotations), medianRotation, std::end(rotations));

    auto mtx = *medianTx;
    auto mty = *medianTy;
    auto mtr = *medianRotation;

    _pTargetDataset->transform = agg::trans_affine(cos(mtr), -sin(mtr), sin(mtr), cos(mtr), -mtx, -mty);
}

bool Aligner::CheckTransform()
{
    uint32_t matches = 0;

    for (uint32_t i = 0; i < _pTargetDataset->stars.size(); ++i)
    {
        auto targetPoint = _pTargetDataset->stars[i].center;
        _pTargetDataset->transform.transform(&targetPoint.x, &targetPoint.y);

        for (uint32_t j = 0; j < _pRefDataset->stars.size(); ++j)
        {
            auto refPoint = _pRefDataset->stars[j].center;
            auto dist = refPoint.Distance(targetPoint);
            //BUGBUG magic number
            if (dist < 5)
            {
                ++matches;
                break;
            }
        }
    }

    if (matches > 10)
    {
        std::cout << "match count = " << matches << std::endl;
    }

    return (matches > 10);
}

bool Aligner::TryRefPair(const std::pair<Star, Star>& refPair)
{
    std::pair<PointF, PointF> refPoints{refPair.first.center, refPair.second.center};
    auto refDist = refPoints.first.Distance(refPoints.second);

    std::pair<Star, Star> targetPair { Star{}, Star{} };
    std::pair<PointF, PointF> targetPoints{ PointF{}, PointF{} };
    for (uint32_t i = 0; i < _pTargetDataset->stars.size() - 1; ++i)
    {
        targetPair.first = _pTargetDataset->stars[i];
        targetPoints.first = targetPair.first.center;

        for (uint32_t j = i + 1; j < _pTargetDataset->stars.size(); ++j)
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
