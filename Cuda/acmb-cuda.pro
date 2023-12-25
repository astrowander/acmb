QT       -= core gui

TARGET = acmb-cuda
TEMPLATE = lib
CONFIG += staticlib
CONFIG += c++2a

LIBS += -L$$OUT_PWD/../ -lacmb-lib

INCLUDEPATH += $$PWD/../
DEPENDPATH += $$PWD/../

PRE_TARGETDEPS += $$OUT_PWD/../libacmb-lib.a

INCLUDEPATH += /usr/local/include
DEPENDPATH += /usr/local/include

HEADERS += \
    AddBitmap.h \
    AddBitmapWithAlignment.cuh \
    AddBitmapWithAlignment.h \
    CudaBasic.h \
    CudaBasic.hpp \
    CudaStacker.h \
    GenerateResult.h\
    CudaInfo.h

SOURCES += \
    AddBitmapWithAlignment.cpp \
    CudaStacker.cpp\
    CudaInfo.cpp

CUDA_SOURCES += \
    AddBitmap.cu\
    AddBitmapWithAlignment.cu\
    GenerateResult.cu

CUDA_DIR = /usr/lib/cuda
# GPU architecture
#https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#virtual-architecture-feature-list
CUDA_VARCH = compute_87
#https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list-dc -rdc=true
CUDA_GPU_ARCH = sm_87

# nvcc flags (ptxas option verbose is always useful)
NVCCFLAGS = --compiler-options -use-fast-math --Wno-deprecated-gpu-targets
# include paths
INCLUDEPATH += $$CUDA_DIR/include
#INCLUDEPATH += /opt/cub/cub         #Поддержка заголовочных файлов библиотеки CUB (Более производительная и низкоуровневая замена Thrust)
# include paths
#INCLUDEPATH += /usr/local/cuda/include
# lib dirs
QMAKE_LIBDIR += $$CUDA_DIR/lib64
# libs - note than i'm using a x_86_64 machine
LIBS += -lcuda -lcudart -lcurand
# join the includes in a line
CUDA_INC = $$join(INCLUDEPATH,' -I','-I',' ')

# Prepare the extra compiler configuration (taken from the nvidia forum - i'm not an expert in this part)
cuda.input = CUDA_SOURCES
cuda.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.o

#Улучшенный вывод сообещний об ошибках. Более понятный для QtCreator, чтобы можно было по двойному щелчку переходить к строке в файле

#Параметр -dc __device__ функцию используемую __kernel__ функцией в одном файле, можно объявить/определить в другом файле
#-dc == --device-c -> Compile each .c, .cc, .cpp, .cxx, and .cu input file into an object file that contains relocatable device code.
# -rdc=true|false == relocatable device code
#-dc is equivalent to --relocatable-device-code=true
#--compiler-options '-fPIC' -dc −rdc=true

#Если добавить -x, то nvcc будет воспринимать .cpp файлы, как .cu (содержащие cuda-код).
#Должен будет более адекватно работать QtCreator, эти .cpp файлы всё равно надо будет происывать в CUDA_SOURCES.
#Но для наглядности лучше оставить их .cu и эту опцию не использовать. Иначе можно запутаться.
CONFIG(debug, debug|release) {
# NEED: cuda.commands = $$CUDA_DIR/bin/nvcc -m64 -g -G -gencode arch=$$CUDA_VARCH,code=$$CUDA_GPU_ARCH -c -dc -rdc=true $$NVCCFLAGS \   !!!!!!!!!!!!!!!!!!!!!
cuda.commands = $$CUDA_DIR/bin/nvcc -m64 -g -G -gencode arch=$$CUDA_VARCH,code=$$CUDA_GPU_ARCH -c $$NVCCFLAGS \
                $$CUDA_INC $$LIBS  ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT} \
                2>&1 | sed -r \"s/\\(([0-9]+)\\)/:\\1/g\" 1>&2
}
else {
cuda.commands = $$CUDA_DIR/bin/nvcc -m64 -O3 -gencode arch=$$CUDA_VARCH,code=$$CUDA_GPU_ARCH -c $$NVCCFLAGS \
                $$CUDA_INC $$LIBS  ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT} \
                2>&1 | sed -r \"s/\\(([0-9]+)\\)/:\\1/g\" 1>&2
}

#Оригинал с GitHub
#cuda.commands = $$CUDA_DIR/bin/nvcc -m64 -g -G -gencode arch=$$CUDA_VARCH,code=$$CUDA_GPU_ARCH -c $$NVCCFLAGS $$CUDA_INC $$LIBS  ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}

cuda.dependency_type = TYPE_C # there was a typo here. Thanks workmate!

CONFIG(debug, debug|release) {
    cuda.depend_command = $$CUDA_DIR/bin/nvcc -g -G -M $$CUDA_INC $$NVCCFLAGS   ${QMAKE_FILE_NAME}  | sed \"s/^.*: //\"
}
else {
    cuda.depend_command = $$CUDA_DIR/bin/nvcc -O3 -M $$CUDA_INC $$NVCCFLAGS   ${QMAKE_FILE_NAME}  | sed \"s/^.*: //\"
}

# Tell Qt that we want add more stuff to the Makefile
QMAKE_EXTRA_UNIX_COMPILERS += cuda

