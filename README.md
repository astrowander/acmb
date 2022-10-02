# What is it? 
acmb is an open-source cross-platform software for astronomical photographic image manipulation.
## Building instructions
### Download sources from GitHub
```sh
$ git clone https://github.com/astrowander/Astrocombine
```
### Preparing Third Parties
Create subdirectory Libs in the main project folder
```sh
$ cd Astrocombine && mkdir Libs && cd Libs
```
Download the following libraries from GitHub
```sh
$ git clone https://github.com/LibRaw/LibRaw
$ git clone https://github.com/lensfun/lensfun
$ git clone https://github.com/oneapi-src/oneTBB
$ git clone https://github.com/jkriege2/TinyTIFF
```
Build and install them using their own instructions

### Building acmb on Windows
Open the solution acmb.sln with Visual Studio 2022, and build it
### Building acmb on Linux
Open the project file acmb.pro with QtCreator, and build it. Please note, that you need a compiler with support of C++20

## Downloading Binaries
You can also download binaries for x64 Windows and x64 Linux. Linux binary requires of installed third party libraries (please see above)

##How to use it
Open in the console the directory where the executable is located (or add it to PATH environment variable)
### Input and output
Firstly you must specify input and output files, otherwise you'll got an error. For example
```sh
$ acmb --input "/path/to/input/file" --output "/path/to/output/file"
```
You can specify either a single input file or a directory. Also you can define range of input files like that:
```sh
$ acmb --input "/path/to/input/IMG_1349#1353.CR2" --output "/path/to/output/file"
```
This means you will load five files: IMG_1349.CR2, IMG_1350.CR2, IMG_1351.CR2, IMG_1352.CR2, IMG_1353.CR2. Also if you have several inputs you must specify output directory, not the single file.
