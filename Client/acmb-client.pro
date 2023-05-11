TEMPLATE = app
CONFIG += console c++2a
CONFIG -= app_bundle
CONFIG -= qt

LIBS += -lboost_system

SOURCES += \
    CliParser.cpp \
    client.cpp \
    main.cpp \
    tools.cpp

INCLUDEPATH += /usr/local/include
DEPENDPATH += /usr/local/include

HEADERS += \
    CliParser.h \
    client.h \
    enums.h \
    tools.h
