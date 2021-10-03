TEMPLATE = app
CONFIG += console c++17
CONFIG -= app_bundle
CONFIG -= qt

LIBS += -lraw

SOURCES += \
        Registrator/registrator.cpp \
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
    Core/bitmap.h \
    Core/imageparams.h \
    Core/enums.h \
    Registrator/registrator.h \
    Transforms/converter.h \
    Codecs/imagedecoder.h \
    Codecs/imageencoder.h \
    Codecs/PPM/ppmdecoder.h \
    Codecs/PPM/ppmencoder.h \
    Tests/test.h \
    Tests/testtools.h
