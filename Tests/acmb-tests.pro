TEMPLATE = app
CONFIG += console c++2a
CONFIG -= app_bundle
CONFIG -= qt

LIBS += -L/usr/local/lib/ -lraw
LIBS += -llensfun
LIBS += -lTinyTIFFShared_Release
LIBS += -ltbb
LIBS += -L/usr/local/lib/ -lCCfits
LIBS += -L/usr/local/lib/ -lcfitsio
LIBS += -lx265
LIBS += -lcudart

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
    TestConverter.cpp \
    TestCropTransform.cpp \
    TestCudaStacker.cpp \
    TestDeaberrateTransform.cpp \
    TestDebayerTransform.cpp \
    TestFastAligner.cpp \
    TestFitsDecoder.cpp \
    TestFitsEncoder.cpp \
    TestH265Encoder.cpp \
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
    TestResizeTransform.cpp \
    TestRgbToHsl.cpp \
    TestRunner.cpp \
    TestSaturationTransform.cpp \
    TestStacker.cpp \
    TestStarTrekTransform.cpp \
    TestTiffDecoder.cpp \
    TestTiffEncoder.cpp \
    TestY4MEncoder.cpp \
    main.cpp \
    testtools.cpp

LIBS += -L$$OUT_PWD/../ -lacmb-lib
LIBS += -L$$OUT_PWD/../Cuda/ -lacmb-cuda

INCLUDEPATH += $$PWD/../
DEPENDPATH += $$PWD/../

PRE_TARGETDEPS += $$OUT_PWD/../libacmb-lib.a
PRE_TARGETDEPS += $$OUT_PWD/../Cuda/libacmb-cuda.a

INCLUDEPATH += /usr/local/include
DEPENDPATH += /usr/local/include
