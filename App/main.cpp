#include "CliParser.h"
#include "../Cuda/CudaStacker.h"
#include "../Cuda/CudaInfo.h"
#include "../Registrator/stacker.h"

#include "../Codecs/SER/SerDecoder.h"
#include "../Registrator/SortingHat.h"
#include "../Registrator/FastDetector.h"
#include "../Transforms/converter.h"
#include "../Transforms/BinningTransform.h"
#include "../Transforms/ResizeTransform.h"
#include "../Transforms/MedianBlurTransform.h"
#include "../Transforms/LaplacianTransform.h"
#include "../Transforms/HistogramBuilder.h"
#include "../Transforms/ChannelEqualizer.h"
#include <chrono>

using namespace acmb;
int main( int argc, const char** argv )
{
    try
    {
        auto start = std::chrono::steady_clock::now();
        /*auto [res, errMsg] = acmb::CliParser::Parse(argc, argv,
        [] ( const std::vector<acmb::Pipeline>& pipelines, acmb::StackMode stackMode, bool enableCudaIfAvailable ) -> std::shared_ptr<acmb::BaseStacker>
        {
            if ( acmb::cuda::isCudaAvailable() && enableCudaIfAvailable )
            {
                std::cout << "CUDA is enabled" << std::endl;
                return std::make_shared<acmb::cuda::Stacker>( pipelines, stackMode );
            }

            return std::make_shared<acmb::Stacker>( pipelines, stackMode );
        });
        if ( !errMsg.empty() )
            std::cout << errMsg << std::endl;*/

            /*auto pDecoder = ImageDecoder::Create("F:/Images/jupiter.ser", PixelFormat::RGB48);
            SortingHat sortingHat( *pDecoder );

            for ( int i = 0; i < pDecoder->GetFrameCount(); ++i )
            {
                if ( i % 100 == 0 )
                    std::cout << i << std::endl;

                auto pBitmap = pDecoder->ReadBitmap();
                pBitmap = MedianBlurTransform::MedianBlur( pBitmap, 3 );
                //pBitmap = ResizeTransform::Resize( pBitmap, { 60, 60 } );
                auto pBinningTransform = BinningTransform::Create( pBitmap, { 5,5 } );
                pBitmap = pBinningTransform->RunAndGetBitmap();
                sortingHat.AddFrame( pBitmap );
            }
            std::cout << std::endl;

            const float qualityThreshold = 0.75f;
            auto bestFrames = sortingHat.GetBestFramesByQualityThreshold( qualityThreshold );
            std::cout << bestFrames.size() << "frames found with " << qualityThreshold << " quality threshold" << std::endl;
            const auto& frames = sortingHat.Frames();
            auto bestIt = frames.begin();
            std::cout << "Best: " << std::endl;
            std::cout << "score: " << bestIt->first << std::endl;
            auto pBestBitmap = pDecoder->ReadBitmap( bestIt->second.index );*/
        auto pBestBitmap = IBitmap::Create( "D:/Images/jupiter_best.tif", PixelFormat::RGB48 );
        pBestBitmap = Converter::Convert( pBestBitmap, PixelFormat::Gray16 );
        auto pBinningTransform = BinningTransform::Create( pBestBitmap, { 5,5 } );
        pBestBitmap = pBinningTransform->RunAndGetBitmap();

        const auto pixelFormat = pBestBitmap->GetPixelFormat();
        const auto bytesPerChannel = BytesPerChannel( pixelFormat );
        const float absoluteMax = (bytesPerChannel == 1) ? 255.0f : 65535.0f;

        auto pHistogramBuilder = HistogramBuilder::Create( pBestBitmap );
        pHistogramBuilder->BuildHistogram();
        //constexpr float logTargetMedian = 1.0f;

        const float minLevel = pHistogramBuilder->GetChannelStatistics( 0 ).min / absoluteMax;
        const float maxLevel = pHistogramBuilder->GetChannelStatistics( 0 ).max / absoluteMax;
        //const float denom = log( (pHistogramBuilder->GetChannelStatistics( 0 ).median / absoluteMax- minLevel) / (maxLevel - minLevel) );
        const float gamma = 1.5f;

        pBestBitmap = ChannelEqualizer::Equalize( pBestBitmap,
                                                  {
                                                        [&]( float srcVal )
                                                        {
                                                            if ( srcVal < minLevel )
                                                                return 0.0f;
                                                            if ( srcVal > maxLevel )
                                                                return 1.0f;

                                                            float res = (srcVal - minLevel) / (maxLevel - minLevel);
                                                            res = std::pow( res, gamma );

                                                            return res;
                                                        }
                                                  } );
        //IBitmap::Save( pBestBitmap, "F:/Images/jupiter_best_downscaled.tif" );
        //IBitmap::Save( bestIt->second.pBitmap, "F:/Images/jupiter_best_laplacian.tif" );
        IBitmap::Save( pBestBitmap, "D:/Images/jupiter_best_downscaled.tif" );
        const auto features = DetectFeatures( pBestBitmap, 0.1f, 6000 );
        //auto pGrayBitmap = std::static_pointer_cast<Bitmap<PixelFormat::Gray16>>(pBestBitmap);
        for ( auto& feature : features )
            pBestBitmap->SetChannel( feature.x, feature.y, 0, 65535 );
        IBitmap::Save( pBestBitmap, "D:/Images/jupiter_best_downscaled_fast.tif" );
        /*auto secondIt = std::next(bestIt);
        std::cout << "Second: " << std::endl;
        std::cout << "score: " << secondIt->first << std::endl;
        IBitmap::Save( pDecoder->ReadBitmap( secondIt->second.index ), "F:/Images/jupiter_second.tif" );
        IBitmap::Save( secondIt->second.pBitmap, "F:/Images/jupiter_second_laplacian.tif" );

        auto worstIt = frames.rbegin();
        std::cout << "Worst: " << std::endl;
        std::cout << "score" << worstIt->first << std::endl;
        IBitmap::Save( pDecoder->ReadBitmap( worstIt->second.index ), "F:/Images/jupiter_worst.tif" );

        //pResult = MedianBlurTransform::MedianBlur( pResult, 3 );
        //IBitmap::Save( pResult, "F:/Images/jupiter_laplacian.tif" );

       /* SerDecoder decoder;
        decoder.Attach( "F:\\Projects\\AstroCombine\\Tests\\TestFiles\\SER\\19_45_36_crop.ser" );

        auto frameCount = decoder.GetFrameCount();
        std::vector<IBitmapPtr> bitmaps( frameCount );
        for ( size_t i = 0; i < frameCount; ++i )
        {
            bitmaps[i] = decoder.ReadBitmap();
            bitmaps[i] = Converter::Convert( bitmaps[i], PixelFormat::Gray16 );
        }

        SortingHat sortingHat( bitmaps );
        sortingHat.SortAndFilter();*/

        auto duration = std::chrono::steady_clock::now() - start;
        const size_t totalMilliSecs = std::chrono::duration_cast< std::chrono::milliseconds >(duration).count();
        std::cout << "Elapsed " << totalMilliSecs / 1000 << " s " << totalMilliSecs % 1000 << " ms" << std::endl;

        return 0;
    }
    catch ( std::exception& e )
    {
        std::cout << e.what() << std::endl;
    }

    return 1;
}
