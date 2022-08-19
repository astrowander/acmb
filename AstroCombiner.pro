TEMPLATE = app
CONFIG += console c++2a
CONFIG -= app_bundle
CONFIG -= qt

LIBS += -lraw
LIBS += -llensfun

SOURCES += \
        AGG/agg_trans_affine.cpp \
        Codecs/Raw/RawDecoder.cpp \
        Core/enums.cpp \
        Geometry/startrektransform.cpp \
        Geometry/triangle.cpp \
        Registrator/FastAligner.cpp \
        Registrator/registrator.cpp \
        Registrator/stacker.cpp \
        Tests/TestBinningTransform.cpp \
        Tests/TestBitmap.cpp \
        Tests/TestBitmapSubtractor.cpp \
        Tests/TestChannelEqualizer.cpp \
        Tests/TestCliParser.cpp \
        Tests/TestConverter.cpp \
        Tests/TestFastAligner.cpp \
        Tests/TestHaloRemoval.cpp \
        Tests/TestHistogramBuilder.cpp \
        Tests/TestNewton2D.cpp \
        Tests/TestPpmDecoder.cpp \
        Tests/TestPpmEncoder.cpp \
        Tests/TestRawDecoder.cpp \
        Tests/TestRegistrator.cpp \
        Tests/TestRgbToHsl.cpp \
        Tests/TestRunner.cpp \
        Tests/TestStacker.cpp \
        Tests/TestStarTrekTransform.cpp \
        Tests/testtools.cpp \
        Core/bitmap.cpp \
        Core/imageparams.cpp \
        Tools/CliParser.cpp \
        Tools/Newton2D.cpp \
        Tools/mathtools.cpp \
        Transforms/BitmapSubtractor.cpp \
        Transforms/ChannelEqualizer.cpp \
        Transforms/HaloRemovalTransform.cpp \
        Transforms/HistogramBuilder.cpp \
        Transforms/basetransform.cpp \
        Transforms/binningtransform.cpp \
        Transforms/converter.cpp \
        Codecs/imagedecoder.cpp \
        Codecs/imageencoder.cpp \
        Codecs/PPM/ppmdecoder.cpp \
        Codecs/PPM/ppmencoder.cpp\
        Transforms/deaberratetransform.cpp \
        main.cpp


HEADERS += \
    AGG/agg_basics.h \
    AGG/agg_config.h \
    AGG/agg_trans_affine.h \
    Codecs/Raw/RawDecoder.h \
    Core/IParallel.h \
    Core/bitmap.h \
    Core/camerasettings.h \
    Core/imageparams.h \
    Core/enums.h \
    Geometry/delaunator.hpp \
    Geometry/point.h \
    Geometry/rect.h \
    Geometry/size.h \
    Geometry/startrektransform.h \
    Geometry/triangle.h \
    Registrator/AddingBitmapHelper.h \
    Registrator/AddingBitmapWithAlignmentHelper.h \
    Registrator/AlignmentHelper.h \
    Registrator/FastAligner.h \
    Registrator/GeneratingResultHelper.h \
    Registrator/alignmentdataset.h \
    Registrator/registrator.h \
    Registrator/stacker.h \
    Registrator/star.h \
    Tests/TestRunner.h \
    Tools/CliParser.h \
    Tools/Newton2D.h \
    Tools/mathtools.h \
    Transforms/BitmapSubtractor.h \
    Transforms/ChannelEqualizer.h \
    Transforms/HaloRemovalTransform.h \
    Transforms/HistogramBuilder.h \
    Transforms/basetransform.h \
    Transforms/binningtransform.h \
    Transforms/converter.h \
    Codecs/imagedecoder.h \
    Codecs/imageencoder.h \
    Codecs/PPM/ppmdecoder.h \
    Codecs/PPM/ppmencoder.h \
    Tests/test.h \
    Tests/testtools.h \
    Transforms/deaberratetransform.h
