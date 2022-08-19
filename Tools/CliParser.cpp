#include "CliParser.h"

#include "./../Tests/TestRunner.h"
#include "./../Registrator/stacker.h"
#include "./../Transforms/ChannelEqualizer.h"
#include "./../Transforms/HaloRemovalTransform.h"
#include <filesystem>

CliParser::CliParser( int argc, const char** argv )
{
    if ( argc < 1 )
        throw std::invalid_argument( "argc" );

    if ( !argv )
        throw std::invalid_argument( "argv" );

    for ( int i = 0; i < argc; )
    {
        if ( argv[i][0] != '-' )
        {
            ++i;
            continue;
        }

        if ( i == argc - 1 || argv[i + 1][0] == '-' )
        {
            _kv.insert( { argv[i], "" } );
            ++i;
            continue;
        }

        _kv.insert( { argv[i], argv[i + 1] } );
        i += 2;
    }
}

std::tuple<int, std::string> CliParser::Parse( bool testMode )
{
    auto it = _kv.find( "-runtests" );
    if ( !testMode && it != std::end( _kv ) )
    {
        it = _kv.find( "-suite" );
        if ( it == std::end( _kv ) )
        {
            TestRunner::RunAllTests();
            return {};
        }

        std::string suite = it->second;

        it = _kv.find( "-test" );

        if ( it == std::end( _kv ) )
        {
            TestRunner::RunSuite( suite );
            return {};
        }

        TestRunner::RunTest( suite, it->second );
        return {};
    }

    it = _kv.find( "-stack" );
    if ( it != std::end( _kv ) )
    {
        std::string paths;
        it = _kv.find( "-input" );
        if ( it == std::end( _kv ) )
        {
            return { 1, "Input files are not specified" };
        }
        paths = it->second;

        it = _kv.find( "-output" );

        std::string outputPath;
        it = _kv.find( "-output" );
        if ( it == std::end( _kv ) )
        {            
            return { 1, "Output file is not specified" };
        }

        outputPath = it->second; 

        _darks.clear();
        it = _kv.find( "-darks" );
        if ( it != std::end( _kv ) )
        {
            _darks = ImageDecoder::GetDecodersFromMask( it->second );
        }

        _decoders.clear();
        _decoders = ImageDecoder::GetDecodersFromMask( paths ); 

        if ( _decoders.empty() )
        {
            return { 1, "Nothing to stack" };
        }

        if ( testMode )
            return {};

        Stacker stacker( _decoders );
        if ( !_darks.empty() )
        {
            Stacker darkStacker( _darks );
            stacker.SetDarkFrame( darkStacker.Stack( false ) );
        }
        double threshold = 40;
        it = _kv.find( "-threshold" );
        if ( it != std::end(_kv) )
        {
            threshold = std::stod( it->second );
        }

        auto pRes = (_kv.find("-noalign") == std::end(_kv)) ? stacker.RegistrateAndStack( threshold ) : stacker.Stack(false);

        it = _kv.find( "-autowb" );
        if ( it != std::end( _kv ) )
        {
            pRes = BaseChannelEqualizer::AutoEqualize( pRes );
        }

        it = _kv.find( "-removehalo" );
        if ( it != std::end( _kv ) )
        {
            float intensity = 1.0f;
            if ( !it->second.empty() )
            {
                intensity = std::stof( it->second ) / 100.0f;
            }
            pRes = BaseHaloRemovalTransform::AutoRemove( pRes, intensity );
        }
        IBitmap::Save( pRes, outputPath );
        return {};
    }

    return { 1, "Nothing to do" };
}

std::tuple<int, std::string> CliParser::Parse( int argc, const char** argv )
{
    CliParser parser(argc, argv);
    return parser.Parse();
}
