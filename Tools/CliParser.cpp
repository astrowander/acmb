#include "CliParser.h"

#include "./../Core/versioning.h"
#include "./../Codecs/imagedecoder.h"
#include "./../Codecs/imageencoder.h"
#include "./../Registrator/stacker.h"
#include "./../Transforms/binningtransform.h"
#include "./../Transforms/BitmapSubtractor.h"
#include "./../Transforms/BitmapDivisor.h"
#include "./../Transforms/converter.h"
#include "./../Transforms/ChannelEqualizer.h"
#include "./../Transforms/deaberratetransform.h"
#include "./../Transforms/DebayerTransform.h"
#include "./../Transforms/HaloRemovalTransform.h"
#include "./../Transforms/ResizeTransform.h"
#include "./../Transforms/CropTransform.h"
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
        return {0, std::string("acmb version ") + FULL_VERSION };

    if ( _kvs.front().key != "--input" || _kvs.front().values.empty() )
        return { 1, "Input files must be specified in the first place" };

    if ( _kvs.back().key != "--output" || _kvs.back().values.empty() )
        return { 1, "Output files must be specified in the last place" };

    size_t iStart = 1;
    PixelFormat desiredFormat = PixelFormat::Unspecified;
    if ( _kvs[1].key == "--desiredFormat" )
    {
        iStart = 2;
        const auto& values = _kvs[1].values;

        if ( stringToPixelFormat.contains( values[0]) )
        {            
            desiredFormat = stringToPixelFormat.at( values[0]);
        }
    }

    for ( const auto& inputString : _kvs.front().values )
    {
        auto pipelines = ImageDecoder::GetPipelinesFromMask( inputString, desiredFormat );
        if ( pipelines.empty() )
            return { 1, "no input files" };

        _pipelinesBeforeStacker.insert( _pipelinesBeforeStacker.end(), pipelines.begin(), pipelines.end() );
    }

    bool isStackerFound = false;

    for ( size_t i = iStart; i < _kvs.size() - 1; ++i )
    {
        const auto& key = _kvs[i].key;
        const auto& values = _kvs[i].values;

        if ( key == "--binning" )
        {
            if ( values.size() != 2 )
                return { 1, "--binning requires exactly two arguments" };

            const int width = std::stoi( values[0] );
            const int height = std::stoi( values[1] );
            if ( width <= 0 || height <= 0 )
            {
                return { 1, "--binning requires strictly positive arguments" };
            }
            Size bin{ uint32_t( width ), uint32_t( height ) };
            if ( isStackerFound )
            {
                _pipelineAfterStacker.AddTransform<BinningTransform>( bin );
            }
            else
            {
                for ( auto& pipeline : _pipelinesBeforeStacker )
                {
                    pipeline.AddTransform<BinningTransform>( bin );
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
                _pipelineAfterStacker.AddTransform<Converter>( it->second );
            }
            else
            {
                for ( auto& pipeline : _pipelinesBeforeStacker )
                {
                    pipeline.AddTransform<Converter>( it->second );
                }
            }
        }
        else if ( key == "--subtract" )
        {
            if ( values.size() != 1 )
                return { 1, "--subtract requires exactly one argument" };

            auto pBitmapToSubtract = IBitmap::Create( values[0], ( ( isStackerFound ) ? _pipelineAfterStacker.GetFinalParams()->GetPixelFormat() : _pipelinesBeforeStacker[0].GetFinalParams()->GetPixelFormat() ) );

            if ( isStackerFound )
            {
                _pipelineAfterStacker.AddTransform<BitmapSubtractor>( pBitmapToSubtract );
            }
            else
            {
                for ( auto& pipeline : _pipelinesBeforeStacker )
                {
                    pipeline.AddTransform<BitmapSubtractor>( pBitmapToSubtract );
                }
            }
        }
        else if ( key == "--divide" )
        {
            if ( values.size() < 1 || values.size() > 2 )
                return { 1, "--divide requires one or two arguments" };

            auto pDivisor = IBitmap::Create( values[0], ( (isStackerFound) ? _pipelineAfterStacker.GetFinalParams()->GetPixelFormat() : _pipelinesBeforeStacker[0].GetFinalParams()->GetPixelFormat() ) );
            float intensity = 100.0f;
            if ( values.size() == 2 )
            {
                intensity = std::stof( values[1] );
            }

            if ( isStackerFound )
            {
                _pipelineAfterStacker.AddTransform<BitmapDivisor>( { pDivisor, intensity } );
            }
            else
            {
                for ( auto& pipeline : _pipelinesBeforeStacker )
                {
                    pipeline.AddTransform<BitmapDivisor>( { pDivisor, intensity } );
                }
            }
        }
        else if ( key == "--autowb" )
        {
            if ( values.size() != 0 )
                return { 1, "--subtract requires no argument" };

            if ( isStackerFound )
            {
                _pipelineAfterStacker.AddTransform<AutoChannelEqualizer>();
            }
            else
            {
                for ( auto& pipeline : _pipelinesBeforeStacker )
                {                   
                    pipeline.AddTransform<AutoChannelEqualizer>();
                }
            }
        }
        else if ( key == "--deaberrate" )
        {
            if ( values.size() != 0 )
                return { 1, "--deaberrate requires no argument" };

            if ( isStackerFound )
            {
                _pipelineAfterStacker.AddTransform<DeaberrateTransform>( _pipelineAfterStacker.GetCameraSettings() );
            }
            else
            {
                for ( auto& pipeline : _pipelinesBeforeStacker )
                {
                    pipeline.AddTransform<DeaberrateTransform>( _pipelineAfterStacker.GetCameraSettings() );
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
                _pipelineAfterStacker.AddTransform<AutoHaloRemoval>( intensity );
            }
            else
            {
                for ( auto& pipeline : _pipelinesBeforeStacker )
                {
                    pipeline.AddTransform<AutoHaloRemoval>( intensity );
                }
            }
        }
        else if ( key == "--resize" )
        {
        if ( values.size() != 2 )
            return { 1, "--resize requires exactly 2 arguments" };

        const auto width = uint32_t(std::stoi( values[0] ));
        const auto height = uint32_t(std::stoi( values[1] ));

        if ( isStackerFound )
        {
            _pipelineAfterStacker.AddTransform<ResizeTransform>( Size {width, height });
        }
        else
        {
            for ( auto& pipeline : _pipelinesBeforeStacker )
            {
                pipeline.AddTransform<ResizeTransform>( Size{ width, height } );
            }
        }
        }
        else if ( key == "--crop" )
        {
        if ( values.size() != 4 )
            return { 1, "--crop requires exactly 4 arguments" };

        const int x = std::stoi( values[0] );
        const int y = std::stoi( values[1] );
        const int width = std::stoi( values[2] );
        const int height = std::stoi( values[3] );

        if ( isStackerFound )
        {
            _pipelineAfterStacker.AddTransform<CropTransform>( Rect { x, y, width, height });
        }
        else
        {
            for ( auto& pipeline : _pipelinesBeforeStacker )
            {
                pipeline.AddTransform<CropTransform>( Rect{ x, y, width, height } );
            }
        }
        }
        else if ( key == "--stack" )
        {
        if ( isStackerFound )
            return { 1, "only one --stack is allowed" };

        isStackerFound = true;

        std::shared_ptr<Stacker> pStacker;
        if ( values.empty() || values[0] == "light" )
            pStacker = std::make_shared<Stacker>( _pipelinesBeforeStacker, StackMode::Light );
        else if ( values[0] == "dark" || values[0] == "flat" )
            pStacker = std::make_shared<Stacker>( _pipelinesBeforeStacker, StackMode::DarkOrFlat );
        else
            return { 1, "invalid stack mode" };

        _pipelineAfterStacker.Add( pStacker );
        }
        else if ( key == "--debayer" )
        {
        if ( values.size() != 0 )
            return { 1, "--deaberrate requires no argument" };

        if ( isStackerFound )
        {
            _pipelineAfterStacker.AddTransform<DebayerTransform>( _pipelineAfterStacker.GetCameraSettings() );
        }
        else
        {
            for ( auto& pipeline : _pipelinesBeforeStacker )
            {
                pipeline.AddTransform<DebayerTransform>( _pipelineAfterStacker.GetCameraSettings() );
            }
        }
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
