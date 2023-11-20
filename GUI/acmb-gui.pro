TEMPLATE = app
TARGET = acmb-gui2
CONFIG += console c++2a
CONFIG -= app_bundle
CONFIG -= qt

LIBS += -L/usr/local/lib/ -lraw
LIBS += -llensfun
LIBS += -lTinyTIFFShared_Release
LIBS += -ltbb
LIBS += -lcudart
LIBS += -lglfw
LIBS += -lvulkan

LIBS += -L$$OUT_PWD/../ -lacmb-lib
LIBS += -L$$OUT_PWD/../Cuda/ -lacmb-cuda

INCLUDEPATH += $$PWD/../
DEPENDPATH += $$PWD/../

PRE_TARGETDEPS += $$OUT_PWD/../libacmb-lib.a
PRE_TARGETDEPS += $$OUT_PWD/../Cuda/libacmb-cuda.a

INCLUDEPATH += /usr/local/include
DEPENDPATH += /usr/local/include

INCLUDEPATH += "./imgui/"\
                "./imgui/backends/"
SOURCES += \
        ConverterWindow.cpp \
        CropWindow.cpp \
        FileDialog.cpp \
        FlatFieldWindow.cpp \
        FontRegistry.cpp \
        ImGuiFileDialog/ImGuiFileDialog.cpp \
        ImGuiHelpers.cpp \
        ImageReaderWindow.cpp \
        ImageWriterWindow.cpp \
        MainWindow.cpp \
        MenuItem.cpp \
        MenuItemsHolder.cpp \
        PipelineElementWindow.cpp \
        ResizeWindow.cpp \
        StackerWindow.cpp \
        SubtractImageWindow.cpp \
        imgui/backends/imgui_impl_glfw.cpp \
        imgui/backends/imgui_impl_vulkan.cpp \
        imgui/imgui.cpp \
        imgui/imgui_demo.cpp \
        imgui/imgui_draw.cpp \
        imgui/imgui_tables.cpp \
        imgui/imgui_widgets.cpp \
        main.cpp \
        window.cpp

HEADERS += \
    ConverterWindow.h \
    CropWindow.h \
    FileDialog.h \
    FlatFieldWindow.h \
    FontRegistry.h \
    ImGuiFileDialog/ImGuiFileDialog.h \
    ImGuiFileDialog/ImGuiFileDialogConfig.h \
    ImGuiFileDialog/dirent/dirent.h \
    ImGuiFileDialog/stb/stb_image.h \
    ImGuiFileDialog/stb/stb_image_resize.h \
    ImGuiHelpers.h \
    ImageReaderWindow.h \
    ImageWriterWindow.h \
    MainWindow.h \
    MenuItem.h \
    MenuItemsHolder.h \
    PipelineElementWindow.h \
    ResizeWindow.h \
    Serializer.h \
    StackerWindow.h \
    SubtractImageWindow.h \
    imgui/backends/imgui_impl_glfw.h \
    imgui/backends/imgui_impl_vulkan.h \
    imgui/imconfig.h \
    imgui/imgui.h \
    imgui/imgui_internal.h \
    imgui/imstb_rectpack.h \
    imgui/imstb_textedit.h \
    imgui/imstb_truetype.h \
    tl/expected.hpp \
    window.h
