TEMPLATE = app
CONFIG += console c++2a
CONFIG -= app_bundle
CONFIG -= qt

LIBS += -L/usr/local/lib/ -lraw
LIBS += -llensfun
LIBS += -lTinyTIFFShared_Release
LIBS += -ltbb

HEADERS += \
    TestRunner.h \
    test.h \
    testtools.h

SOURCES += \
    TestBinningTransform.cpp \
    TestBitmap.cpp \
    TestBitmapDivisor.cpp \
    TestBitmapSubtractor.cpp \
    TestChannelEqualizer.cpp \
    TestCliParser.cpp \
    TestConverter.cpp \
    TestDeaberrateTransform.cpp \
    TestDebayerTransform.cpp \
    TestFastAligner.cpp \
    TestHaloRemoval.cpp \
    TestHistogramBuilder.cpp \
    TestImageDecoder.cpp \
    TestImageEncoder.cpp \
    TestJpegEncoder.cpp \
    TestNewton2D.cpp \
    TestPipeline.cpp \
    TestPpmDecoder.cpp \
    TestPpmEncoder.cpp \
    TestRawDecoder.cpp \
    TestRegistrator.cpp \
    TestRgbToHsl.cpp \
    TestRunner.cpp \
    TestStacker.cpp \
    TestStarTrekTransform.cpp \
    TestTiffDecoder.cpp \
    TestTiffEncoder.cpp \
    main.cpp \
    testtools.cpp

LIBS += -L$$OUT_PWD/../ -lacmb-lib

INCLUDEPATH += $$PWD/../
DEPENDPATH += $$PWD/../

PRE_TARGETDEPS += $$OUT_PWD/../libacmb-lib.a

INCLUDEPATH += /usr/local/include
DEPENDPATH += /usr/local/include
