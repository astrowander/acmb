#include "CliParser.h"

#include "./../Tests/TestRunner.h"
#include "./../Registrator/stacker.h"
#include <filesystem>

std::string IntToString( int num, int minDigitCount )
{
    auto res = std::to_string( num );
    if ( res.size() < minDigitCount )
    {
        res.insert( 0, std::string( minDigitCount - res.size(), '0' ) );
    }

    return res;
}

CliParser::CliParser( int argc, char** argv )
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

int CliParser::Parse( bool testMode )
{
    auto it = _kv.find( "-runtests" );
    if ( !testMode && it != std::end( _kv ) )
    {
        it = _kv.find( "-suite" );
        if ( it == std::end( _kv ) )
        {
            TestRunner::GetInstance().RunAllTests();
            return 0;
        }

        std::string suite = it->second;

        it = _kv.find( "-test" );

        if ( it == std::end( _kv ) )
        {
            TestRunner::GetInstance().RunAllTestsInSuite( suite );
            return 0;
        }

        TestRunner::GetInstance().RunTest( suite, it->second );
        return 0;
    }

    it = _kv.find( "-stack" );
    if ( it != std::end( _kv ) )
    {
        auto paths = it->second;

        std::string outputPath;
        it = _kv.find( "-output" );
        if ( it == std::end( _kv ) )
        {
            std::cout << "Output file is not specified" << std::endl;
            return 1;
        }

        outputPath = it->second;
        

        it = _kv.find( "-dir" );
        if ( it != std::end( _kv ) )
        {
            if ( !std::filesystem::is_directory( it->second ) )
            {
                std::cout << "No such directory" << std::endl;
            }

            for ( const auto& entry : std::filesystem::directory_iterator( it->second ) )
            {
                if ( !std::filesystem::is_directory( entry ) )
                    _decoders.push_back( ImageDecoder::Create( entry.path().u8string() ) );
            }
        }
        else
        {
            size_t start = 0;

            while ( start < paths.size() )
            {
                auto end = paths.find_first_of( ';', start );
                std::string fileName = paths.substr( start, end - start );
                size_t tildePos = fileName.find_first_of( '~' );

                if ( std::filesystem::exists( fileName ) )
                {
                    _decoders.push_back( ImageDecoder::Create( fileName ) );
                }
                else if ( tildePos != std::string::npos )
                {
                    size_t pointPos = fileName.find_first_of( '.', tildePos );
                    auto varDigitCount = pointPos - tildePos - 1;
                    if ( varDigitCount != 0 )
                    {

                        int minNum = std::stoi( fileName.substr( tildePos - varDigitCount, varDigitCount ) );
                        int maxNum = std::stoi( fileName.substr( tildePos + 1, varDigitCount ) );

                        for ( int j = minNum; j <= maxNum; ++j )
                        {
                            auto tempName = fileName.substr( 0, tildePos - varDigitCount ) + IntToString( j, varDigitCount ) + fileName.substr( pointPos );
                            if ( std::filesystem::exists( tempName ) )
                                _decoders.push_back( ImageDecoder::Create( tempName ) );
                        }
                    }
                }

                if ( end == std::string::npos )
                    break;

                start = end + 1;
            }
        }

        if ( _decoders.empty() )
        {
            std::cout << "Nothing to stack" << std::endl;
            return 1;
        }

        if ( testMode )
            return 0;

        Stacker stacker( _decoders );
        auto pRes = stacker.RegistrateAndStack( 9, 6 );
        IBitmap::Save( pRes, outputPath );
    }
    return 0;
}

int CliParser::Parse( int argc, char** argv )
{
    CliParser parser(argc, argv);
    return parser.Parse();
}
