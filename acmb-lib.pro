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

SOURCES += \
    AGG/agg_trans_affine.cpp \
    Codecs/imageencoder.cpp \
    Codecs/imagedecoder.cpp \
    Codecs/JPEG/toojpeg/toojpeg.cpp \
    Codecs/JPEG/JpegEncoder.cpp \
    Codecs/PPM/ppmdecoder.cpp \
    Codecs/PPM/ppmencoder.cpp \
    Codecs/Raw/RawDecoder.cpp \
    Codecs/Tiff/TiffDecoder.cpp \
    Codecs/Tiff/TiffEncoder.cpp \
    Core/bitmap.cpp \
    Core/enums.cpp \
    Core/imageparams.cpp \
    Geometry/startrektransform.cpp \
    Geometry/triangle.cpp \
    Registrator/FastAligner.cpp \
    Registrator/registrator.cpp \
    Registrator/stacker.cpp \
    Tools/CliParser.cpp \
    Tools/mathtools.cpp \
    Tools/Newton2D.cpp \
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
    Codecs/imageencoder.h \
    Codecs/imagedecoder.h \
    Codecs/JPEG/toojpeg/toojpeg.h \
    Codecs/JPEG/JpegEncoder.h \
    Codecs/PPM/ppmdecoder.h \
    Codecs/PPM/ppmencoder.h \
    Codecs/Raw/RawDecoder.h \
    Codecs/Tiff/TiffDecoder.h \
    Codecs/Tiff/TiffEncoder.h \
    Core/bitmap.h \
    Core/camerasettings.h \
    Core/enums.h \
    Core/imageparams.h \
    Geometry/delaunator.hpp \
    Geometry/point.h \
    Geometry/rect.h \
    Geometry/size.h \
    Geometry/startrektransform.h \
    Geometry/triangle.h \
    Registrator/FastAligner.h \
    Registrator/registrator.h \
    Registrator/stacker.h \
    Registrator/star.h \
    Tools/CliParser.h \
    Tools/mathtools.h \
    Tools/Newton2D.h \
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
