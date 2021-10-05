TEMPLATE = app
CONFIG += console c++17
CONFIG -= app_bundle
CONFIG -= qt

LIBS += -lraw

SOURCES += \
        AGG/agg_trans_affine.cpp \
        Geometry/matrix.cpp \
        Geometry/rect.cpp \
        Registrator/aligner.cpp \
        Registrator/registrator.cpp \
        Tests/TestAligner.cpp \
        Tests/TestBitmap.cpp \
        Tests/TestConverter.cpp \
        Tests/TestPpmDecoder.cpp \
        Tests/TestPpmEncoder.cpp \
        Tests/TestRegistrator.cpp \
        Tests/testtools.cpp \
        Core/bitmap.cpp \
        Core/imageparams.cpp \
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
    Geometry/matrix.h \
    Geometry/rect.h \
    Registrator/aligner.h \
    Registrator/alignmentdataset.h \
    Registrator/registrator.h \
    Registrator/star.h \
    Transforms/converter.h \
    Codecs/imagedecoder.h \
    Codecs/imageencoder.h \
    Codecs/PPM/ppmdecoder.h \
    Codecs/PPM/ppmencoder.h \
    Tests/test.h \
    Tests/testtools.h
