#include "CliParser.h"

#include "./../Codecs/imagedecoder.h"
#include "./../Codecs/imageencoder.h"
#include "./../Registrator/stacker.h"
#include "./../Transforms/BinningTransform.h"
#include "./../Transforms/BitmapSubtractor.h"
#include "./../Transforms/Converter.h"
#include "./../Transforms/ChannelEqualizer.h"
#include "./../Transforms/DeaberrateTransform.h"
#include "./../Transforms/HaloRemovalTransform.h"
#include "./../Tools/SystemTools.h"
#include <filesystem>


ACMB_NAMESPACE_BEGIN

static const std::unordered_map<std::string, PixelFormat> stringToPixelFormat =
{
    {"gray8", PixelFormat::Gray8},
    {"gray16", PixelFormat::Gray16},
    {"rgb24", PixelFormat::RGB24},
    {"rgb48", PixelFormat::RGB48}
};


CliParser::CliParser( int argc, const char** argv )
{
    if ( argc < 1 )
        throw std::invalid_argument( "argc" );

    if ( !argv )
        throw std::invalid_argument( "argv" );

    for ( int i = 1; i < argc; ++i )
    {
        if ( argv[i][0] == '-' && argv[i][1] == '-' )
        {
            _kvs.push_back( { argv[i], {} } );
            continue;
        }

        if ( _kvs.empty() )
            throw std::invalid_argument( "no key" );

        auto& kv = _kvs.back();
        kv.values.push_back( argv[i] );
    }
}

std::tuple<int, std::string> CliParser::Parse( bool testMode )
{
    if ( _kvs.empty() )
        return { 1, "Nothing to do" };

    if ( _kvs.front().key != "--input" || _kvs.front().values.empty() )
        return { 1, "Input files must be specified in the first place" };

    if ( _kvs.back().key != "--output" || _kvs.back().values.empty() )
        return { 1, "Output files must be specified in the last place" };

    for ( const auto& inputString : _kvs.front().values )
    {
        auto pipelines = ImageDecoder::GetPipelinesFromMask( inputString );
        _pipelinesBeforeStacker.insert( _pipelinesBeforeStacker.end(), pipelines.begin(), pipelines.end() );
    }

    bool isStackerFound = false;

    for ( size_t i = 1; i < _kvs.size() - 1; ++i )
    {
        const auto& key = _kvs[i].key;
        const auto& values = _kvs[i].values;

        if ( key == "--binning" )
        {
            if ( values.size() != 2 )
                return { 1, "--binning requires exactly two arguments" };

            Size bin{ std::stoi( values[0] ), std::stoi( values[1] ) };
            if ( isStackerFound )
            {
                auto pBinningTransform = BinningTransform::Create( _pipelineAfterStacker.GetFinalParams()->GetPixelFormat(), bin );
                _pipelineAfterStacker.Add( pBinningTransform );
            }
            else
            {
                for ( auto& pipeline : _pipelinesBeforeStacker )
                {
                    auto pBinningTransform = BinningTransform::Create( pipeline.GetFinalParams()->GetPixelFormat(), bin );
                    pipeline.Add( pBinningTransform );
                }
            }
        }
        else if ( key == "--convert" )
        {
            if ( values.size() != 1 )
                return { 1, "--convert requires exactly one argument" };

            auto it = stringToPixelFormat.find( ToLower( values[0] ) );
            if ( it == std::end( stringToPixelFormat ) )
                return { 1, "wrong pixel format" };

            if ( isStackerFound )
            {
                auto pConverter = Converter::Create( _pipelineAfterStacker.GetFinalParams()->GetPixelFormat(), it->second );
                _pipelineAfterStacker.Add( pConverter );
            }
            else
            {
                for ( auto& pipeline : _pipelinesBeforeStacker )
                {
                    auto pConverter = Converter::Create( pipeline.GetFinalParams()->GetPixelFormat(), it->second );
                    pipeline.Add( pConverter );
                }
            }
        }
        else if ( key == "--subtract" )
        {
            if ( values.size() != 1 )
                return { 1, "--subtract requires exactly one argument" };

            auto pBitmapToSubtract = IBitmap::Create( values[0] );

            if ( isStackerFound )
            {
                auto pConverter = BitmapSubtractor::Create( _pipelineAfterStacker.GetFinalParams()->GetPixelFormat(), pBitmapToSubtract );
                _pipelineAfterStacker.Add( pConverter );
            }
            else
            {
                for ( auto& pipeline : _pipelinesBeforeStacker )
                {
                    auto pConverter = BitmapSubtractor::Create( pipeline.GetFinalParams()->GetPixelFormat(), pBitmapToSubtract );
                    pipeline.Add( pConverter );
                }
            }
        }
        else if ( key == "--autowb" )
        {
            if ( values.size() != 0 )
                return { 1, "--subtract requires no argument" };

            if ( isStackerFound )
            {
                _pipelineAfterStacker.Add( std::make_shared<AutoChannelEqualizer>() );
            }
            else
            {
                for ( auto& pipeline : _pipelinesBeforeStacker )
                {                   
                    pipeline.Add( std::make_shared<AutoChannelEqualizer>() );
                }
            }
        }
        else if ( key == "--deaberrate" )
        {
            if ( values.size() != 0 )
                return { 1, "--deaberrate requires no argument" };

            if ( isStackerFound )
            {
                auto pDeaberrateTransform = DeaberrateTransform::Create( _pipelineAfterStacker.GetFinalParams()->GetPixelFormat(), _pipelineAfterStacker.GetCameraSettings() );
                _pipelineAfterStacker.Add( pDeaberrateTransform );
            }
            else
            {
                for ( auto& pipeline : _pipelinesBeforeStacker )
                {
                    auto pDeaberrateTransform = DeaberrateTransform::Create( pipeline.GetFinalParams()->GetPixelFormat(), pipeline.GetCameraSettings() );
                    pipeline.Add( pDeaberrateTransform );
                }
            }
        }
        else if ( key == "--removehalo" )
        {
            float intensity = 75.0f;
            if ( values.size() > 1 )
                return { 1, "--subtract requires 0 or 1 argument" };
            else if ( values.size() == 1 )
                intensity = std::stof( values[0] );

            if ( isStackerFound )
            {
                _pipelineAfterStacker.Add( std::make_shared<AutoHaloRemoval>(nullptr, intensity ) );
            }
            else
            {
                for ( auto& pipeline : _pipelinesBeforeStacker )
                {
                    pipeline.Add( std::make_shared<AutoHaloRemoval>( nullptr, intensity ) );
                }
            }
        }
        else if ( key == "--stack" )
        {
        if ( isStackerFound )
            return { 1, "only one --stack is allowed" };        

        isStackerFound = true;

        auto pStacker = std::make_shared<Stacker>( _pipelinesBeforeStacker );
        if ( !values.empty() && values[0] == "noalign" )
            pStacker->SetDoAlignment( false );

        _pipelineAfterStacker.Add( pStacker );
        }
        else
        {
            return { 1, "unknown key" };
        }
    }

    if ( testMode )
        return {};

    const std::string& pathToOutput = _kvs.back().values[0];    

    if ( isStackerFound )
    {
        _pipelineAfterStacker.Add( ImageEncoder::Create( pathToOutput ) );
    }
    else if ( std::filesystem::is_directory( pathToOutput ) )
    {
        for ( auto& pipeline : _pipelinesBeforeStacker )
        {
            const auto fullPath = pipeline.GetFileName();
            const auto fileName = fullPath.substr( fullPath.find_last_of( "/\\" ) );
            pipeline.Add( ImageEncoder::Create( pathToOutput + "/" + fileName  + "." + ( ( _kvs.back().values.size() > 0 ) ? _kvs.back().values[1] : "tif" ) ) );
        }
    }
    else
    {
        if ( _pipelinesBeforeStacker.size() != 1 )
            return { 1, "There are several inputs. You must specify output directory, not the single file" };

        _pipelinesBeforeStacker[0].Add( ImageEncoder::Create( pathToOutput ) );
    }

    

    if ( isStackerFound )
    {
        _pipelineAfterStacker.RunAndGetBitmap();
        return {};
    }
    
    for ( auto& pipeline : _pipelinesBeforeStacker )
    {
        std::cout << pipeline.GetFileName() << std::endl;
        pipeline.RunAndGetBitmap();
    }

    return {};
}

std::tuple<int, std::string> CliParser::Parse( int argc, const char** argv )
{
    CliParser parser(argc, argv);
    return parser.Parse();
}

ACMB_NAMESPACE_END
