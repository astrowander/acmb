TEMPLATE = app
CONFIG += console c++17
CONFIG -= app_bundle
CONFIG -= qt

LIBS += -lraw

SOURCES += \
        Tests/TestBitmap.cpp \
        Tests/TestConverter.cpp \
        Tests/TestPpmDecoder.cpp \
        Tests/TestPpmEncoder.cpp \
        Tests/testtools.cpp \
        Core/bitmap.cpp \
        Core/imageparams.cpp \
        Transforms/converter.cpp \
        Codecs/imagedecoder.cpp \
        Codecs/imageencoder.cpp \
        main.cpp \
        Codecs/PPM/ppmdecoder.cpp \
        Codecs/PPM/ppmencoder.cpp


HEADERS += \
    Core/bitmap.h \
    Core/imageparams.h \
    Core/enums.h \
    Transforms/converter.h \
    Codecs/imagedecoder.h \
    Codecs/imageencoder.h \
    Codecs/PPM/ppmdecoder.h \
    Codecs/PPM/ppmencoder.h \
    Tests/test.h \
    Tests/testtools.h
