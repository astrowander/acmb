TEMPLATE = app
CONFIG += console c++17
CONFIG -= app_bundle
CONFIG -= qt

LIBS += -lraw

SOURCES += \
        TestBitmap.cpp \
        TestConverter.cpp \
        TestPpmDecoder.cpp \
        TestPpmEncoder.cpp \
        bitmap.cpp \
        converter.cpp \
        imagedecoder.cpp \
        imageencoder.cpp \
        imageparams.cpp \
        main.cpp \
        ppmdecoder.cpp \
        ppmencoder.cpp \
        testtools.cpp

HEADERS += \
    bitmap.h \
    converter.h \
    enums.h \
    imagedecoder.h \
    imageencoder.h \
    imageparams.h \
    ppmdecoder.h \
    ppmencoder.h \
    test.h \
    testtools.h
