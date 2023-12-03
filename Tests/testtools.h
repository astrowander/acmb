#pragma once
#include "./../Core/macros.h"
#include "./../Core/bitmap.h"
#include "./../Codecs/PPM/ppmencoder.h"

#include <memory>
#include <string>
#include <filesystem>

ACMB_NAMESPACE_BEGIN
class IBitmap;
ACMB_NAMESPACE_END

ACMB_TESTS_NAMESPACE_BEGIN

bool BitmapsAreEqual(std::shared_ptr<IBitmap> lhs, std::shared_ptr<IBitmap> rhs);
bool BitmapsAreEqual(const std::string& fileName, std::shared_ptr<IBitmap> rhs);

std::string GetPathToTestFile(const std::string& fileName);
std::string GetPathToPattern(const std::string& fileName);

template<class DecoderType>
bool TestPixelFormat( const std::string& pixelFormat )
{
    auto pRefBitmap = IBitmap::Create( GetPathToTestFile( std::string( "TIFF/" ) + pixelFormat + ".tiff" ) );
    DecoderType encoder;
    auto tempDir = GetPathToTestFile( "/tmp/" );

    PpmEncoder ppmEncoder( PpmMode::Binary );
    std::string ppmFileName = tempDir + pixelFormat + "_temp.ppm";
    ppmEncoder.Attach( ppmFileName );
    ppmEncoder.WriteBitmap( pRefBitmap );
    ppmEncoder.Detach();

    std::string tmpFileName = tempDir + pixelFormat + "_temp" + *DecoderType::GetExtensions().begin();

    encoder.Attach( tmpFileName );
    encoder.WriteBitmap( pRefBitmap );
    encoder.Detach();

    auto pTargetBitmap = IBitmap::Create( tmpFileName );
    const bool res = BitmapsAreEqual( pRefBitmap, pTargetBitmap );

    std::filesystem::remove( ppmFileName );
    std::filesystem::remove( tmpFileName );
    return res;
}

ACMB_TESTS_NAMESPACE_END