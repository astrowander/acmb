#-------------------------------------------------
#
# Project created by QtCreator 2022-09-16T09:27:29
#
#-------------------------------------------------

QT       -= core gui

TARGET = acmb-lib
TEMPLATE = lib
CONFIG += staticlib
CONFIG += c++2a
INCLUDEPATH += Libs/cfitsio/include

SOURCES += \
    AGG/agg_trans_affine.cpp \
    Codecs/FITS/FitsDecoder.cpp \
    Codecs/FITS/FitsEncoder.cpp \
    Codecs/H265/H265Encoder.cpp \
    Codecs/JPEG/JpegDecoder.cpp \
    Codecs/JPEG/TJpg_Decoder/tjpgd.cpp \
    Codecs/VideoEncoder.cpp \
    Codecs/Y4M/Y4MEncoder.cpp \
    Codecs/imageencoder.cpp \
    Codecs/imagedecoder.cpp \
    Codecs/JPEG/toojpeg/toojpeg.cpp \
    Codecs/JPEG/JpegEncoder.cpp \
    Codecs/PPM/ppmdecoder.cpp \
    Codecs/PPM/ppmencoder.cpp \
    Codecs/Raw/RawDecoder.cpp \
    Codecs/Tiff/TiffDecoder.cpp \
    Codecs/Tiff/TiffEncoder.cpp \
    Core/IPipelineElement.cpp \
    Core/bitmap.cpp \
    Core/enums.cpp \
    Core/imageparams.cpp \
    Core/log.cpp \
    Core/pipeline.cpp \
    Geometry/startrektransform.cpp \
    Geometry/triangle.cpp \
    Registrator/BaseStacker.cpp \
    Registrator/FastAligner.cpp \
    Registrator/registrator.cpp \
    Registrator/stacker.cpp \
    Tools/SystemTools.cpp \
    Tools/mathtools.cpp \
    Tools/Newton2D.cpp \
    Transforms/BitmapDivisor.cpp \
    Transforms/CropTransform.cpp \
    Transforms/DebayerTransform.cpp \
    Transforms/ResizeTransform.cpp \
    Transforms/SaturationTransform.cpp \
    Transforms/basetransform.cpp \
    Transforms/binningtransform.cpp \
    Transforms/BitmapSubtractor.cpp \
    Transforms/ChannelEqualizer.cpp \
    Transforms/converter.cpp \
    Transforms/deaberratetransform.cpp \
    Transforms/HaloRemovalTransform.cpp \
    Transforms/HistogramBuilder.cpp

HEADERS += \
    AGG/agg_trans_affine.h \
    AGG/agg_config.h \
    AGG/agg_basics.h \
    Codecs/FITS/FitsDecoder.h \
    Codecs/FITS/FitsEncoder.h \
    Codecs/H265/H265Encoder.h \
    Codecs/JPEG/JpegDecoder.h \
    Codecs/JPEG/TJpg_Decoder/tjpgd.h \
    Codecs/JPEG/TJpg_Decoder/tjpgdcnf.h \
    Codecs/Raw/RawSettings.h \
    Codecs/VideoEncoder.h \
    Codecs/Y4M/Y4MEncoder.h \
    Codecs/imageencoder.h \
    Codecs/imagedecoder.h \
    Codecs/JPEG/toojpeg/toojpeg.h \
    Codecs/JPEG/JpegEncoder.h \
    Codecs/PPM/ppmdecoder.h \
    Codecs/PPM/ppmencoder.h \
    Codecs/Raw/RawDecoder.h \
    Codecs/Tiff/TiffDecoder.h \
    Codecs/Tiff/TiffEncoder.h \
    Core/IPipelineElement.h \
    Core/bitmap.h \
    Core/camerasettings.h \
    Core/enums.h \
    Core/imageparams.h \
    Core/log.h \
    Core/macros.h \
    Core/pipeline.h \
    Core/versioning.h \
    Geometry/delaunator.hpp \
    Geometry/point.h \
    Geometry/rect.h \
    Geometry/size.h \
    Geometry/startrektransform.h \
    Geometry/triangle.h \
    Registrator/BaseStacker.h \
    Registrator/FastAligner.h \
    Registrator/StackEngineConstants.h \
    Registrator/registrator.h \
    Registrator/stacker.h \
    Registrator/star.h \
    Tools/SystemTools.h \
    Tools/mathtools.h \
    Tools/Newton2D.h \
    Transforms/BitmapDivisor.h \
    Transforms/CropTransform.h \
    Transforms/DebayerTransform.h \
    Transforms/ResizeTransform.h \
    Transforms/SaturationTransform.h \
    Transforms/basetransform.h \
    Transforms/binningtransform.h \
    Transforms/BitmapSubtractor.h \
    Transforms/ChannelEqualizer.h \
    Transforms/converter.h \
    Transforms/deaberratetransform.h \
    Transforms/HaloRemovalTransform.h \
    Transforms/HistogramBuilder.h
unix {
    target.path = /usr/lib
    INSTALLS += target
}
