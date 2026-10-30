TEMPLATE = app
TARGET = acmb-gui-tests
CONFIG += console c++2a
CONFIG -= app_bundle qt

INCLUDEPATH += $$PWD/../GUI/imgui

HEADERS += \
    TestAppHost.h \
    ../GUI/MainWindowLayout.h \
    ../GUI/PipelineEditing.h \
    ../GUI/GuiInteractions.h \
    ../GUI/ScrollablePanels.h \
    ../GUI/TopToolbar.h \
    ../GUI/ToolLibrary.h \
    ../GUI/StepPresentation.h \
    ../Tests/TestRunner.h \
    ../Tests/test.h \
    ../GUI/MenuItem.h

SOURCES += \
    main.cpp \
    TestGuiSmoke.cpp \
    TestLayoutInputs.cpp \
    TestMainWindowPanels.cpp \
    TestPipelineEditing.cpp \
    TestGuiInteractions.cpp \
    TestScrollablePanels.cpp \
    TestTopToolbar.cpp \
    TestToolLibrary.cpp \
    TestStepPresentation.cpp \
    ../Tests/TestRunner.cpp \
    ../GUI/imgui/imgui.cpp \
    ../GUI/imgui/imgui_draw.cpp \
    ../GUI/imgui/imgui_tables.cpp \
    ../GUI/imgui/imgui_widgets.cpp
