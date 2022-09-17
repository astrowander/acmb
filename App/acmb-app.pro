TEMPLATE = app
CONFIG += console c++2a
CONFIG -= app_bundle
CONFIG -= qt

LIBS += -L/usr/local/lib/ -lraw
LIBS += -llensfun
LIBS += -lTinyTIFFShared_Release
LIBS += -ltbb

SOURCES += \
    main.cpp \

LIBS += -L$$OUT_PWD/../ -lacmb-lib

INCLUDEPATH += $$PWD/../
DEPENDPATH += $$PWD/../

PRE_TARGETDEPS += $$OUT_PWD/../libacmb-lib.a

INCLUDEPATH += /usr/local/include
DEPENDPATH += /usr/local/include
