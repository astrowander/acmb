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
#include "../Transforms/DeconvTransform.h"
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

        const std::string filePath = "F:\\Projects\\AstroCombine\\Tests\\TestFiles\\TIFF\\m22.tif";
        auto pDecoder = ImageDecoder::Create(filePath, PixelFormat::RGB48);
        auto pBitmap = pDecoder->ReadBitmap();

        auto pDeconvTransform = DeconvTransform::Create(pBitmap, { 1.0 });
        pBitmap = pDeconvTransform->RunAndGetBitmap();

        pBitmap->Save(pBitmap, "F:\\Projects\\AstroCombine\\Tests\\TestFiles\\TIFF\\m22_deconv.tif");



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
