#include "DeconvTransform.h"
#include "converter.h"
#include "../Registrator/registrator.h"
#include "../Geometry/delaunator.hpp"
#include "../Geometry/triangle.h"

#include <optional>
#include <algorithm>
#include <unordered_map>


ACMB_NAMESPACE_BEGIN

static constexpr double epsilon = 1e-6;

struct StarMoments
{
    double theta;
    double L;
    double sigma;

    auto operator<=>(const StarMoments& rhs) const = default;
};

struct StarMomentsHasher
{
    std::size_t operator()(const StarMoments& moments) const
    {
        std::size_t h1 = std::hash<double>{}(moments.theta);
        std::size_t h2 = std::hash<double>{}(moments.L);
        std::size_t h3 = std::hash<double>{}(moments.sigma);
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

static std::optional<StarMoments> CalculateStarMoments(const Star::MomentMatrix& m)
{
    const double trace = m.xx + m.yy;
    const double d = sqrt((m.xx - m.yy) * (m.xx - m.yy) + 4 * m.xy * m.xy);
    const double lambdaMax = (trace + d) / 2.0;
    const double lambdaMin = (trace - d) / 2.0;

    if ( lambdaMin <= 0 || lambdaMax <= 0 )
        return std::nullopt;

    double L2 = 12.0 * (lambdaMax - lambdaMin);
    if ( L2 < 0.0 )
        return std::nullopt;    

    StarMoments res { .theta = 0.5 * atan2(2 * m.xy, m.xx - m.yy),
                      .L = sqrt(L2),
                      .sigma = sqrt(lambdaMin) };

    res.L = round(res.L * 100.0) / 100.0;
    res.sigma = round(res.sigma * 10.0) / 10.0;
    res.theta = round(res.theta * 10.0) / 10.0;

    return res;
}

int CalcKernelSize(double sigma, double L)
{
    const double k = 4.0;

    int radius = static_cast<int>(
        std::ceil(0.5 * L + k * sigma)
        );

    return 2 * radius + 1;
}

static std::vector<Star> FindStars(std::shared_ptr<IBitmap> pBitmap, double thresholdPercents, int minStarSize, int maxStarSize)
{
    Registrator registrator(thresholdPercents, minStarSize, maxStarSize);
    registrator.Registrate(pBitmap);
    std::vector<Star> stars;
    for (const auto& tileStars : registrator.GetStars())
    {
        stars.insert(stars.end(), tileStars.begin(), tileStars.end());
    }
    return stars;
}

static void FilterClippedStars(std::vector<Star>& stars, int width, int height)
{
    // Remove clipped stars
    std::erase_if(stars, [](const Star& star)
    {
        if ( star.isClipped )
            return true;

        return false;
    });

    // Remove stars that are too close to the image borders
    std::erase_if(stars, [width, height](const Star& star)
    {
        if ( star.rect.x <= 0 || star.rect.y <= 0 || star.rect.x + star.rect.width >= int(width) || star.rect.y + star.rect.height >= int(height) )
            return true;
        return false;
    });
}

static void FilterStarsWithInvalidMoments(std::vector<Star>& stars, std::unordered_map<PointD, std::pair<Star, StarMoments>, PointDHasher>& starMomentsMap)
{
    // Remove stars with invalid moments
    std::erase_if(stars, [&starMomentsMap](const Star& star)
    {
        auto moments = CalculateStarMoments(star.m);
        if ( moments.has_value() )
        {
            starMomentsMap[star.center.Rounded(epsilon)] = std::make_pair(star, moments.value());
            return false;
        }
        return true;
    });
}

static void FilterStarsTooFarFromMedian(std::vector<Star>& stars, std::unordered_map<PointD, std::pair<Star, StarMoments>, PointDHasher>& starMomentsMap)
{
    // find median L and sigma
    std::vector<double> LValues;
    std::vector<double> sigmaValues;
    for (const auto& star : stars)
    {
        const auto& moments = starMomentsMap[star.center.Rounded(epsilon)];
        LValues.push_back(moments.second.L);
        sigmaValues.push_back(moments.second.sigma);
    }
    std::sort(LValues.begin(), LValues.end());
    std::sort(sigmaValues.begin(), sigmaValues.end());
    const double medianL = LValues[LValues.size() / 2];
    const double medianSigma = sigmaValues[sigmaValues.size() / 2];
    // Remove stars that are too far from the median L and sigma
    std::erase_if(stars, [medianL, medianSigma, &starMomentsMap](const Star& star)
    {
        auto it = starMomentsMap.find(star.center.Rounded(epsilon));
        if ( it != starMomentsMap.end() && (std::abs(it->second.second.L - medianL) > 0.5 * medianL || std::abs(it->second.second.sigma - medianSigma) > 0.5 * medianSigma) )
        {
            starMomentsMap.erase(it);
            return true;
        }
        return false;
    });
}

static void TriangulateStars(const std::vector<Star>& stars, std::vector<Triangle>& triangles, std::vector<PointD>& hullPoints)
{
    std::vector<double> coords;
    for (const auto& star : stars)
    {
        coords.push_back(star.center.x);
        coords.push_back(star.center.y);
    }
    delaunator::Delaunator d(coords);
    size_t e = d.hull_start;
    do
    {
        hullPoints.push_back(PointD{d.coords[2 * e], d.coords[2 * e + 1]});
        e = d.hull_next[e];
    } while ( e != d.hull_start );
    triangles.reserve(d.triangles.size() / 3);
    for (int i = 0; i < int(d.triangles.size()); i += 3)
    {
        triangles.push_back(Triangle{
            PointD{d.coords[2 * d.triangles[i]], d.coords[2 * d.triangles[i] + 1]},
            PointD{d.coords[2 * d.triangles[i + 1]], d.coords[2 * d.triangles[i + 1] + 1]},
            PointD{d.coords[2 * d.triangles[i + 2]], d.coords[2 * d.triangles[i + 2] + 1]}
        });
    }
}

static void RasterizeTriangles(const std::vector<Triangle>& triangles, int width, int height, std::vector<int>& triangleIndicesMap)
{
    triangleIndicesMap.resize(width * height, -1);
    for (int i = 0; i < int(triangles.size()); ++i)
    {
        const Triangle& tri = triangles[i];
        const auto bbox = tri.GetBoundingBox();
        for (int y = std::max(0, int(bbox.y)); y < std::min(int(height), int(bbox.y + bbox.height)); ++y)
        {
            for (int x = std::max(0, int(bbox.x)); x < std::min(int(width), int(bbox.x + bbox.width)); ++x)
            {
                if (tri.IsPointInsideUnchecked(PointD{double(x + 0.5), double(y + 0.5)}))
                {
                    triangleIndicesMap[y * width + x] = i;
                }
            }
        }
    }
}

struct ProjectionToHullResult
{
    PointD p1;
    PointD p2;
    double param;
};

static ProjectionToHullResult ProjectPointToHull(const PointD& point, const std::vector<PointD>& hullPoints)
{
    double nearestEdgeDistSq = std::numeric_limits<double>::max();
    double nearestParam = 0.0;
    PointD nearestP1;
    PointD nearestP2;
    for (int i = 0; i < int(hullPoints.size()); ++i)
    {
        const PointD& p1 = hullPoints[i];
        const PointD& p2 = hullPoints[(i + 1) % hullPoints.size()];
        auto projectionResult = PointD::ProjectPointOntoSegment(point, p1, p2);
        if (projectionResult.distanceSq < nearestEdgeDistSq)
        {
            nearestEdgeDistSq = projectionResult.distanceSq;
            nearestParam = projectionResult.param;
            nearestP1 = p1;
            nearestP2 = p2;
        }
    }
    return {nearestP1, nearestP2, nearestParam};
}

static Star::MomentMatrix InterpolateMoments(const Star::MomentMatrix& m1, const Star::MomentMatrix& m2, double param)
{
    Star::MomentMatrix m;
    m.xx = (1 - param) * m1.xx + param * m2.xx;
    m.yy = (1 - param) * m1.yy + param * m2.yy;
    m.xy = (1 - param) * m1.xy + param * m2.xy;
    return m;
}

static Star::MomentMatrix InterpolateMoments(const Star::MomentMatrix& m1, const Star::MomentMatrix& m2, const Star::MomentMatrix& m3,
                                          std::array<double, 3> baryCoords)
{
    return Star::MomentMatrix
    {
        .xx = baryCoords[0] * m1.xx + baryCoords[1] * m2.xx + baryCoords[2] * m3.xx,
        .yy = baryCoords[0] * m1.yy + baryCoords[1] * m2.yy + baryCoords[2] * m3.yy,
        .xy = baryCoords[0] * m1.xy + baryCoords[1] * m2.xy + baryCoords[2] * m3.xy
    };
}

struct PSF
{
    int size = 0;
    std::vector<double> data;

    double& operator()(int y, int x)
    {
        return data[y * size + x];
    }

    double operator()(int y, int x) const
    {
        return data[y * size + x];
    }
};

PSF MakePSF(int size, double L, double sigma, double theta)
{
    constexpr double PI = 3.14159265358979323846;
    constexpr double SQRT2 = 1.4142135623730950488;

    sigma = std::max(sigma, 1e-6);
    L = std::max(L, 0.0);

    PSF psf;
    psf.size = size;
    psf.data.assign(size * size, 0.0);

    const double ct = std::cos(theta);
    const double st = std::sin(theta);

    double sum = 0.0;
    const int halfSize = size / 2;

    for ( int ky = 0; ky < size; ++ky )
    {
        for ( int kx = 0; kx < size; ++kx )
        {
            const double x = kx - halfSize;
            const double y = ky - halfSize;

            const double u = x * ct + y * st;
            const double v = -x * st + y * ct;

            double value;

            if ( L < 1e-6 )
            {
                const double r2 = x * x + y * y;

                value =
                    std::exp(-r2 / (2.0 * sigma * sigma)) /
                    (2.0 * PI * sigma * sigma);
            }
            else
            {
                const double a =
                    (u + 0.5 * L) /
                    (SQRT2 * sigma);

                const double b =
                    (u - 0.5 * L) /
                    (SQRT2 * sigma);

                const double longitudinal =
                    std::erf(a) - std::erf(b);

                const double transverse =
                    std::exp(
                        -(v * v) /
                        (2.0 * sigma * sigma)
                    );

                value =
                    transverse * longitudinal /
                    (
                        2.0 *
                        L *
                        std::sqrt(2.0 * PI) *
                        sigma
                        );
            }

            psf(ky, kx) = value;
            sum += value;
        }
    }

    if ( sum > 0.0 )
    {
        for ( double& v : psf.data )
            v /= sum;
    }

    return psf;
}

using PositionToStarAndMomentsMap = std::unordered_map <PointD, std::pair<Star, StarMoments>, PointDHasher>;

class PSFCache
{
    std::unordered_map<StarMoments, int, StarMomentsHasher> _psfIdxCache;
    std::vector<PSF> _psfCache;
    std::vector<int> _psfIndices;
    
public:
    PSFCache(int width, int height)
        : _psfIndices(width * height, -1)
    {
    }

    PSF GetPSF(int px) const
    {
        int psfIdx = _psfIndices[px];
        if ( psfIdx == -1 )
            return PSF{};

        return _psfCache[psfIdx];
    }

    void Clear()
    {
        _psfIdxCache.clear();
        _psfCache.clear();
        std::fill(_psfIndices.begin(), _psfIndices.end(), -1);
    }

    void Generate(const std::vector<Triangle>& triangles, const std::vector<PointD>& hullPoints, const std::vector<int>& triangleIndices, const PositionToStarAndMomentsMap& starMomentsMap, int width, int height)
    {
        Clear();

        for ( int y = 0; y < int(height); ++y )
        {
            for ( int x = 0; x < int(width); ++x )
            {
                Star::MomentMatrix m;

                int triIndex = triangleIndices[y * width + x];
                if ( triIndex == -1 )
                {
                    ProjectionToHullResult projection = ProjectPointToHull(PointD{ x + 0.5, y + 0.5 }, hullPoints);

                    // Interpolate moments from the two nearest hull points
                    auto it1 = starMomentsMap.find(projection.p1.Rounded(epsilon));
                    auto it2 = starMomentsMap.find(projection.p2.Rounded(epsilon));

                    if ( it1 != starMomentsMap.end() && it2 != starMomentsMap.end() )
                    {
                        m = InterpolateMoments(it1->second.first.m, it2->second.first.m, projection.param);
                    }
                }
                else
                {
                    const Triangle& tri = triangles[triIndex];
                    auto baryCoordsOpt = tri.GetBarycentricCoords(PointD{ x + 0.5, y + 0.5 });
                    if ( !baryCoordsOpt.has_value() )
                        continue;

                    auto it1 = starMomentsMap.find(tri.vertices[0].Rounded(epsilon));
                    auto it2 = starMomentsMap.find(tri.vertices[1].Rounded(epsilon));
                    auto it3 = starMomentsMap.find(tri.vertices[2].Rounded(epsilon));

                    if ( it1 == starMomentsMap.end() || it2 == starMomentsMap.end() || it3 == starMomentsMap.end() )
                        continue;

                    m = InterpolateMoments(
                        it1->second.first.m,
                        it2->second.first.m,
                        it3->second.first.m,
                        baryCoordsOpt.value()
                    );
                }

                std::optional<StarMoments> momentsOpt = CalculateStarMoments(m);
                if ( !momentsOpt.has_value() )
                    continue;

                const StarMoments& moments = momentsOpt.value();
                auto psfIt = _psfIdxCache.find(moments);
                int psfIdx = -1;

                if ( psfIt == _psfIdxCache.end() )
                {
                    int kernelSize = CalcKernelSize(moments.sigma, moments.L);
                    PSF psf = MakePSF(kernelSize, moments.L, moments.sigma, moments.theta);

                    psfIdx = static_cast<int>(_psfCache.size());
                    _psfCache.push_back(std::move(psf));
                    _psfIdxCache.insert({ moments, psfIdx });
                }
                else
                {
                    psfIdx = psfIt->second;
                }

                _psfIndices[y * width + x] = psfIdx;
            }
        }
    }
};

template<PixelFormat pixelFormat>
class DeconvTransformImpl : public DeconvTransform
{
    using ChannelType = typename PixelFormatTraits<pixelFormat>::ChannelType;
    static constexpr auto channelCount = PixelFormatTraits<pixelFormat>::channelCount;

    using FPImage = std::vector<double>;

    static FPImage ConvertBitmapToFPImage(std::shared_ptr<Bitmap<pixelFormat>> pBitmap)
    {
        const auto width = pBitmap->GetWidth();
        const auto height = pBitmap->GetHeight();
        FPImage image(width * height * channelCount);
        for (uint32_t y = 0; y < height; ++y)
        {
            for (uint32_t x = 0; x < width; ++x)
            {
                for (uint32_t c = 0; c < channelCount; ++c)
                {
                    image[(y * width + x) * channelCount + c] = static_cast<double>(pBitmap->GetChannel(x, y, c));
                }
            }
        }
        return image;
    }

    static std::shared_ptr<Bitmap<pixelFormat>> ConvertFPImageToBitmap(const FPImage& image, uint32_t width, uint32_t height)
    {
        auto pBitmap = std::make_shared<Bitmap<pixelFormat>>(width, height);
        for (uint32_t y = 0; y < height; ++y)
        {
            for (uint32_t x = 0; x < width; ++x)
            {
                for (uint32_t c = 0; c < channelCount; ++c)
                {
                    double value = image[(y * width + x) * channelCount + c] + 0.5;
                    value = std::clamp(value, 0.0, static_cast<double>(std::numeric_limits<ChannelType>::max()));
                    pBitmap->SetChannel(x, y, c, static_cast<ChannelType>(value));
                }
            }
        }
        return pBitmap;
    }

    static void ApplyH(
        const FPImage& input,
        FPImage& output,
        int width,
        int height,
        const PSFCache& cache)
    {
        output.assign(width * height * channelCount, 0.0);
        Rect bounds(0, 0, width - 1, height - 1);

        for ( int y = 0; y < height; ++y )
        {
            for ( int x = 0; x < width; ++x )
            {
                PSF psf = cache.GetPSF(y * width + x);

                const int r = psf.size / 2;

                for ( int py = 0; py < psf.size; ++py )
                {
                    for ( int px = 0; px < psf.size; ++px )
                    {
                        int xx = x + px - r;
                        int yy = y + py - r;

                        if ( !bounds.IsPointInside(Point(xx, yy)) )
                            continue;

                        for ( uint32_t c = 0; c < channelCount; ++c )
                        {
                            output[(yy * width + xx) * channelCount + c] +=
                                input[(y * width + x) * channelCount + c] * psf(py, px);
                        }
                    }
                }
            }
        }
    }

    static void ApplyHT(
        const FPImage& input,
        FPImage& output,
        int width,
        int height,
        const PSFCache& cache)
    {
        output.assign(width * height * channelCount, 0.0);
        Rect bounds(0, 0, width - 1, height-1);

        for ( int y = 0; y < height; ++y )
        {
            for ( int x = 0; x < width; ++x )
            {
                PSF psf = cache.GetPSF(y * width + x);
                const int r = psf.size / 2;

                std::array<double, channelCount> sum = {};

                for ( int py = 0; py < psf.size; ++py )
                {
                    for ( int px = 0; px < psf.size; ++px )
                    {
                        int xx = x + px - r;
                        int yy = y + py - r;

                        if ( !bounds.IsPointInside(Point(xx, yy)) )
                            continue;

                        for ( uint32_t c = 0; c < channelCount; ++c )
                        { 
                            sum[c] +=
                                input[(yy * width + xx) * channelCount + c] *
                                psf(py, px);
                        }
                    }
                }

                for ( uint32_t c = 0; c < channelCount; ++c )
                {
                    output[(y * width + x) * channelCount + c] = sum[c];
                }
            }
        }
    }

    static std::shared_ptr<Bitmap<pixelFormat>> LucyRichardson(std::shared_ptr<Bitmap<pixelFormat>> pObserved, const PSFCache& cache, int iterations)
    {
        const int width = int(pObserved->GetWidth());
        const int height = int(pObserved->GetHeight());

        FPImage observed = ConvertBitmapToFPImage(pObserved);
        FPImage estimate = observed;
        FPImage predicted;
        FPImage correction;
        FPImage ratio(width * height * channelCount, 0.0);
        

        FPImage ones(width * height * channelCount, 1.0); 
        FPImage normalization;
        ApplyHT(ones, normalization, width, height, cache);

        for ( int iteration = 0; iteration < iterations; ++iteration )
        {
            ApplyH(estimate, predicted, width, height, cache);

            for ( int y = 0; y < height; ++y )
            {
                for ( int x = 0; x < width; ++x )
                {
                    int pixelIndex = (y * width + x) * channelCount;
                    for ( uint32_t c = 0; c < channelCount; ++c )
                    {
                        ratio[pixelIndex + c] = observed[pixelIndex + c] / (predicted[pixelIndex + c] + epsilon);
                    }
                }
            }
        }

        ApplyHT(ratio, correction, width, height, cache);

        for ( int y = 0; y < height; ++y )
        {
            for ( int x = 0; x < width; ++x )
            {
                int pixelIndex = (y * width + x) * channelCount;

                for ( uint32_t c = 0; c < channelCount; ++c )
                {
                    estimate[pixelIndex + c] *=
                        correction[pixelIndex + c] /
                        (normalization[pixelIndex + c] + epsilon);

                    estimate[pixelIndex + c] = std::max(0.0, estimate[pixelIndex + c]);
                }
            }
        }

        return ConvertFPImageToBitmap(estimate, width, height);
    }

public:
    DeconvTransformImpl(std::shared_ptr<Bitmap<pixelFormat>> pSrcBitmap, const Settings& settings)
        : DeconvTransform(pSrcBitmap, settings)
    {
    }

    virtual void Run() override
    {
        const auto width = _pSrcBitmap->GetWidth();
        const auto height = _pSrcBitmap->GetHeight();

        auto pSrcBitmap = std::static_pointer_cast<Bitmap<pixelFormat>>(_pSrcBitmap);
        auto pDstBitmap = pSrcBitmap->Clone();

        constexpr PixelFormat grayFormat = ConstructPixelFormat(PixelFormatTraits<pixelFormat>::bitsPerChannel, 1);
        auto pGraySrcBitmap = std::static_pointer_cast<Bitmap<grayFormat>>(Converter::Convert(pSrcBitmap, grayFormat));
        auto pGrayDstBitmap = std::static_pointer_cast<Bitmap<grayFormat>>(Converter::Convert(pDstBitmap, grayFormat));

        std::vector<Star> stars = FindStars(pGraySrcBitmap, _settings._thresholdPercents, _settings._minStarSize, _settings._maxStarSize);
        FilterClippedStars(stars, width, height);


        PositionToStarAndMomentsMap starMomentsMap;
        FilterStarsWithInvalidMoments(stars, starMomentsMap);

        FilterStarsTooFarFromMedian(stars, starMomentsMap);      

        std::vector<Triangle> triangles;
        std::vector<PointD> hullPoints;
        TriangulateStars(stars, triangles, hullPoints);

        std::vector<int> triangleIndices;
        RasterizeTriangles(triangles, width, height, triangleIndices);

        PSFCache psfCache(width, height);
        psfCache.Generate(triangles, hullPoints, triangleIndices, starMomentsMap, width, height);
       

        _pDstBitmap = LucyRichardson(pSrcBitmap, psfCache, _settings._iterations);
    }

    virtual void ValidateSettings() override
    {
    }
};


DeconvTransform::DeconvTransform(IBitmapPtr pSrcBitmap, const Settings& settings)
    : BaseTransform(pSrcBitmap)
    , _settings(settings)
{
}

std::shared_ptr<DeconvTransform> DeconvTransform::Create(IBitmapPtr pSrcBitmap, const Settings& settings)
{
    switch ( pSrcBitmap->GetPixelFormat() )
    {
        case PixelFormat::RGB24:
            return std::make_shared<DeconvTransformImpl<PixelFormat::RGB24>>(std::static_pointer_cast<Bitmap<PixelFormat::RGB24>>(pSrcBitmap), settings);
        case PixelFormat::RGB48:
            return std::make_shared<DeconvTransformImpl<PixelFormat::RGB48>>(std::static_pointer_cast<Bitmap<PixelFormat::RGB48>>(pSrcBitmap), settings);
        case PixelFormat::Gray8:
            return std::make_shared<DeconvTransformImpl<PixelFormat::Gray8>>(std::static_pointer_cast<Bitmap<PixelFormat::Gray8>>(pSrcBitmap), settings);
        case PixelFormat::Gray16:
            return std::make_shared<DeconvTransformImpl<PixelFormat::Gray16>>(std::static_pointer_cast<Bitmap<PixelFormat::Gray16>>(pSrcBitmap), settings);

        default:
            throw std::invalid_argument("DeconvTransform: Unsupported pixel format");
    }
}

std::shared_ptr<DeconvTransform> DeconvTransform::Create(PixelFormat pixelFormat, const Settings& settings)
{
    switch ( pixelFormat )
    {
        case PixelFormat::RGB24:
            return std::make_shared<DeconvTransformImpl<PixelFormat::RGB24>>(nullptr, settings);
        case PixelFormat::RGB48:
            return std::make_shared<DeconvTransformImpl<PixelFormat::RGB48>>(nullptr, settings);
        case PixelFormat::Gray8:
            return std::make_shared<DeconvTransformImpl<PixelFormat::Gray8>>(nullptr, settings);
        case PixelFormat::Gray16:
            return std::make_shared<DeconvTransformImpl<PixelFormat::Gray16>>(nullptr, settings);

        default:
            throw std::invalid_argument("DeconvTransform: Unsupported pixel format");
    }
}

IBitmapPtr DeconvTransform::ApplyTransform(IBitmapPtr pSrcBitmap, const Settings& settings)
{
    auto pDeconvTransform = Create(pSrcBitmap, settings);
    return pDeconvTransform->RunAndGetBitmap();
}

DeconvTransform::Settings DeconvTransform::Interpolate(const Settings& a, const Settings& b, double t)
{
    Settings result;
    result._intensity = a._intensity + (b._intensity - a._intensity) * t;
    return result;
}

ACMB_NAMESPACE_END