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
![empty_table](https://github.com/astrowander/acmb/assets/7758781/1900d522-486e-4390-8ca3-0f311863d088)
Starting from version 1.0, an GUI version of acmb is available. The principle of operation of this application is similar to Microsoft Excel, with the difference that the cells of the spreadsheet are not numbers and formulas, but images and tools for processing them.
When you open the program, you will see an empty table and a row of buttons above it. To get acquainted with the application, consider as an example a typical astrophotography task. Let's say that we want to combine a set of images of one scene of the starry sky into a single image, optimize it for display on a 4K screen and save it in JPEG format. The original frames were taken with a Canon camera in CR2 format.

### Importing Images
The main frames on which the stars are captured, in fact, are the so-called light frames. We import them first. To do this, select cell A2 and click on the Import button. The Import Images tool will appear in the cell. In it, you need to click the "Select Image" button and select your light frames.

### Subtracting a dark frame
Since each light frame, in addition to useful information, contains the noise of a sensor, it is recommended to take a dark frame with the same exposure and subtract it from the light frame. Select the right adjacent cell B2 with the arrows on the keyboard or by clicking the mouse and add the "Dark Frame" tool to it. Note that cells A2 and B2 are connected by three arrows. This means that the frames between them are transmitted one-by-one, in batch mode.

This tool also requires a dark frame, which we will subtract from the light ones. Since the cell on the left is already occupied, we will put it in the cell on top. To minimize random fluctuations, it is recommended to summarize several dark frames. Add the Import Images tool to cell B1 and upload a series of dark frames. Double-click on the arrows connecting cells B1 and B2. Now they are connected by three converging lines. This means that the images from the top cell will be summed up before they get to the bottom cell.

### Flat field correction
Due to the optical shortcomings of the lens, vignetting or uneven brightness distribution is often found in photographs. To compensate for this effect, you need to photograph a uniformly illuminated field and increase the brightness of the light frame, where the flat frame is darker.
In other words, we have to divide the light frame into a flat field. To do this, place the Flat Field tool in cell C2. By analogy with the previous tool, a series of flat frames must be loaded into cell C1 using the Import Images tool and indicate that they need to be summed up by double-clicking on the arrows.

### Stacking images
After the light frames are calibrated using dark and flat frames, we can put them together into one image. To do this, put the Stacker tool in cell D2. By default, it specifies the "Light Frames" addition mode. This means that the program will search for stars on light frames, combine them with each other so that the stars match and only then sum up the pixel values. The second mode "Dark/Flat Frames" means that the frames will be stacked without alignment, but we need the first mode.

### Changing the size
Let's assume that your camera has a sensor with an aspect ratio of 3:2, and we want to prepare a frame for demonstration on a 4K screen with a resolution of 3840x2160 pixels. It is necessary to resize the image to the required width. To do this, place the Resize tool in cell E2 and specify the values 3840 and 2560 respectively in the Width and Height fields. Such parameters will preserve the aspect ratio of the original frames.

### Cropping
In order for the image to fit completely on the screen, you need to crop it. Add the Crop tool to cell F2 and specify the following parameters: Left = 0, Top = 200, Width = 3840, Height = 2160.

### Changing the color depth
Since we work with RAW frames, our final image has a color depth of 16 bits per pixel. It first needs to be converted to a color depth of 8 bits per pixel, because JPEG does not support a large color depth. Add the Converter tool to cell G2 and select the RGB24 format in it.

### Exporting the result
Now we can save the result of all calculations to disk. Place the Export tool in cell H2. Click the Select File button and specify the JPEG file where you want to save the result of the work.

### Starting calculations
Now that the scheme is ready, you can start the calculations. Click the Run button and wait for the message about the end of the process. Please note that in order to preserve data integrity for the duration of calculations, the entire interface is blocked.

### Saving and loading a project
In order not to create a diagram from scratch every time, you can save it to disk using the Save button and then upload it using the Open button. In the built-in presets catalog, which opens first when the program starts, there are three ready-made schemes: lights.acmb, lights&darks.acmb and lights&darks&flats.acmb for the most common astrophotography processing scenarios.

### Using the GPU
If your computer has an Nvidia graphics adapter with support for CUDA technology, you can enable its support by checking the appropriate box. This can significantly speed up the work of acmb.
![scheme](https://github.com/astrowander/acmb/assets/7758781/ea14bed1-b017-4712-82df-6ed6fd11f4a6)

### acmb in the console
A console version of acmb is also available. You can see the description below.

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
You can combine several pictures to a resulting sum with greater signal-to-noise ratio than a single frame has. By default the application will detect stars in the pictures, search the same stars on the different frames, and align images such a way that stars matched. If you want just to stack images without alignment (e.g. to prepare master dark/flat frame) write argument "dark"/"flat". You can also specify argument "light", it behaves exactly as default.
```sh
$ acmb --input "/path/to/input/files/" --stack [dark,light,flat] --output "/path/to/output/file.tif"
```
### Additional transformations
There are some image transformations in acmb. They can be applied to each input image if you write the key before "--stack" or to stacked result if you write it later. For example:
```sh
$ acmb --input "/path/to/input/files/" --subtract "/path/to/dark/masterdark.tif" --deaberrate --stack [lights] --autowb --removehalo 70 --output "/path/to/output/file.tif"
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
#### --divide "/path/to/image" [intensity]
subtracts given image from their input. Intensity means how strong is effect and changes from 0 (no effect) to 100 (maximal effect)
#### --resize width height
Width and height are the positive integers. This transform resizes the entire image to the given size
#### --crop x y width height
x, y, Width and height are the positive integers. This transform cuts the given rectangular area from the given image. The specified rect must not exceed the image, otherwise you'll get an error.
### Tests
acmb is provided with a bunch of unit tests. If you want to run them please set the environment variable ACMB_PATH="/path/to/acmb/. It must store the location oh the main acmb directory (where you have downloaded it). Also you need to download the set of test files. It is too large to store it in GitHub, you can download it from Google Drive https://drive.google.com/file/d/1whnWlp1ww4q_VxdOZxBYbZuihow6O_T3/view?usp=sharing. Unzip it and place into the "Tests" folder
