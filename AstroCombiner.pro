TEMPLATE = app
CONFIG += console c++17
CONFIG -= app_bundle
CONFIG -= qt

LIBS += -lraw

SOURCES += \
        AGG/agg_trans_affine.cpp \
        Geometry/rect.cpp \
        Registrator/aligner.cpp \
        Registrator/registrator.cpp \
        Registrator/stacker.cpp \
        Tests/TestAligner.cpp \
        Tests/TestBinningTransform.cpp \
        Tests/TestBitmap.cpp \
        Tests/TestConverter.cpp \
        Tests/TestPpmDecoder.cpp \
        Tests/TestPpmEncoder.cpp \
        Tests/TestRegistrator.cpp \
        Tests/TestStacker.cpp \
        Tests/testtools.cpp \
        Core/bitmap.cpp \
        Core/imageparams.cpp \
        Tools/mathtools.cpp \
        Transforms/basetransform.cpp \
        Transforms/converter.cpp \
        Codecs/imagedecoder.cpp \
        Codecs/imageencoder.cpp \
        Codecs/PPM/ppmdecoder.cpp \
        Codecs/PPM/ppmencoder.cpp\
        main.cpp


HEADERS += \
    AGG/agg_basics.h \
    AGG/agg_config.h \
    AGG/agg_trans_affine.h \
    Core/bitmap.h \
    Core/imageparams.h \
    Core/enums.h \
    Geometry/rect.h \
    Registrator/aligner.h \
    Registrator/alignmentdataset.h \
    Registrator/registrator.h \
    Registrator/stacker.h \
    Registrator/star.h \
    Tools/mathtools.h \
    Transforms/basetransform.h \
    Transforms/binningtransform.h \
    Transforms/converter.h \
    Codecs/imagedecoder.h \
    Codecs/imageencoder.h \
    Codecs/PPM/ppmdecoder.h \
    Codecs/PPM/ppmencoder.h \
    Tests/test.h \
    Tests/testtools.h
