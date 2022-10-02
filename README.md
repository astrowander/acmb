# What is it? 
The name "acmb" is a reduction from "AstroCombiner". acmb is an open-source cross-platform software for astronomical photographic image manipulation.
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

## How to use it
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
acmb supports reading from RAW, TIFF and PPM files, and writing to JPEG, TIFF and PPM.
### Stacking
You can combine several pictures to a resulting sum with greater signal-to-noise ration than a single frame has. By default the application will detect stars in the pictures, search the same stars on the different frames, and align images such a way that stars matched. If you want just to stack images without alignment (e.g. to prepare master dark frame) write argument "noalign"
```sh
$ acmb --input "/path/to/input/files/" --stack [noalign] --output "/path/to/output/file.tif"
```
### Additional transformations
There are some image transformations in acmb. They can be applied to each input image if you write the key before "--stack" or to stacked result if you write later. For example:
```sh
$ acmb --input "/path/to/input/files/" --subtract "/path/to/dark/masterdark.tif" --deaberrate --stack [noalign] --autowb --removehalo 70 --output "/path/to/output/file.tif"
```
This command line means: 
1. Load all image files from the input directory
2. Subtract "masterdark.tif" from each image
3. Fix optical aberrations in each image
4. Stack them all to one sum
5. Fix white balance in the sum
6. Remove halos in the sum with 70% intensity
7. Save the result to "file.tif"

See the full list of trasformations below
#### --autowb
It automatically fixes contrast and white balance in the image
#### --binning width height
Width and height are the positive integers. This transform merges each region of given size (bin) in the source image to the single pixel in the destination
#### --convert [gray8, gray16, rgb24, rgb48]
It converts given image to the specified pixel format. Four pixel formats are supported now.
#### --deaberrate
It automatically fixes optical aberrations if the parameners of the lens is known. It uses the EXIF metadata and the database of cameras and lenses from lensfun library.
#### --removehalo intensity
It automatically removes purple halos around bright stars. Intensity means how strong is effect and changes from 0 (no effect) to 100 (maximal effect)
#### --subtract "/path/to/image"
subtracts given image from their input
### Tests
acmb is provided with a bunch of unit tests. If you want to run them please set the environment variable ACMB_PATH="/path/to/acmb/. It must store the location oh the main acmb directory (where you have downloaded it). Also you need to download the set of test files. It is too large to store it in GitHub, you can download it from Google Drive https://drive.google.com/file/d/1whnWlp1ww4q_VxdOZxBYbZuihow6O_T3/view?usp=sharing. Unzip it and place into the "Tests" folder
