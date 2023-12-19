TEMPLATE = app
CONFIG += console c++2a
CONFIG -= app_bundle
CONFIG -= qt

LIBS += -L/usr/local/lib/ -lraw
LIBS += -llensfun
LIBS += -lTinyTIFFShared_Release
LIBS += -ltbb
LIBS += -lcudart
LIBS += -lboost_system
LIBS += -L/usr/local/lib/ -lCCfits
LIBS += -L/usr/local/lib/ -lcfitsio
LIBS += -lx265

SOURCES += \
    main.cpp \
    server.cpp \
    tools.cpp

LIBS += -L$$OUT_PWD/../ -lacmb-lib
LIBS += -L$$OUT_PWD/../Cuda/ -lacmb-cuda

INCLUDEPATH += $$PWD/../
DEPENDPATH += $$PWD/../

PRE_TARGETDEPS += $$OUT_PWD/../libacmb-lib.a
PRE_TARGETDEPS += $$OUT_PWD/../Cuda/libacmb-cuda.a

INCLUDEPATH += /usr/local/include
DEPENDPATH += /usr/local/include

HEADERS += \
    server.h \
    tools.h
