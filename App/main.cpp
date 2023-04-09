#include "../Tools/CliParser.h"
#include "../Cuda/CudaStacker.h"
#include "../Registrator/Stacker.h"

int main(int argc, const char** argv)
{
    try
    {
        auto [res, errMsg] = acmb::CliParser::Parse( argc, argv, 
        [] ( const std::vector<acmb::Pipeline>& pipelines, acmb::StackMode stackMode ) -> std::shared_ptr<acmb::BaseStacker>
        {
            if ( acmb::cuda::isCudaAvailable() )
                return std::make_shared<acmb::cuda::Stacker>( pipelines, stackMode );

            return std::make_shared<acmb::Stacker>( pipelines, stackMode );
        });
        if ( !errMsg.empty() )
            std::cout << errMsg << std::endl;
        return res;
    }
    catch (std::exception& e )
    {
        std::cout << e.what() << std::endl;
    }

    return 1;
}
