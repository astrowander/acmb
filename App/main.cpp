#include "CliParser.h"
#include "../Cuda/CudaStacker.h"
#include "../Registrator/stacker.h"
#include <chrono>

int main(int argc, const char** argv)
{
    try
    {
        auto start = std::chrono::steady_clock::now();
        auto [res, errMsg] = acmb::CliParser::Parse( argc, argv, 
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
            std::cout << errMsg << std::endl;
        auto duration = std::chrono::steady_clock::now() - start;
        const size_t totalMilliSecs = std::chrono::duration_cast< std::chrono::milliseconds >( duration ).count();
        std::cout << "Elapsed " << totalMilliSecs / 1000 << " s " << totalMilliSecs % 1000 << " ms" << std::endl;

        return res;
    }
    catch (std::exception& e )
    {
        std::cout << e.what() << std::endl;
    }

    return 1;
}
