#install packages
sudo apt-get update
sudo apt-get install g++ clang cmake libgtk2.0-dev python3 python3-pip cmake-curses-gui build-essential nasm libglfw3 libglfw3-dev qtcreator qt6-base-dev
pip3 install --upgrade pip3
pip3 install setuptools
#download and install libraries
sudo rm -rf Libs
mkdir Libs
cd Libs
#avir
git clone https://github.com/avaneev/avir
#parallel-hashmap
git clone https://github.com/greg7mdp/parallel-hashmap
#lensfun
git clone https://github.com/lensfun/lensfun
cd lensfun
mkdir build
cd build
cmake ../
make
sudo make install
#tinyTIFF
cd ../../
git clone https://github.com/jkriege2/TinyTIFF
cd TinyTIFF
mkdir build
cd build
cmake .. -DTinyTIFF_BUILD_SHARED_LIBS=ON
cmake --build .
sudo cmake --build . --target install
#cfitsio
cd ../../
git clone https://github.com/healpy/cfitsio
cd cfitsio
./configure --prefix=/usr/local/
make
make install
#CCFits
cd ../
wget https://heasarc.gsfc.nasa.gov/docs/software/fitsio/ccfits/CCfits-2.6.tar.gz
tar -xzvf CCfits-2.6.tar.gz
rm CCfits-2.6.tar.gz
cd CCfits-2.6
./configure --with-cfitsio=/usr/local/
make
sudo make install
#LibRaw
cd ../
git clone https://github.com/LibRaw/LibRaw/
cd LibRaw
autoreconf --install
./configure --disable-lcms --disable-examples
make
sudo make install
#x265
cd ../
git clone https://bitbucket.org/multicoreware/x265_git.git
cd x265_git/build/linux
./make-Makefiles.bash
make
sudo make install
#tbb
cd ../../../
git clone https://github.com/oneapi-src/oneTBB
cd oneTBB
cmake . -DTBB_TEST=OFF
cmake --build .
sudo make install
#cuda
cd ../
mkdir cuda
cd cuda
wget https://developer.download.nvidia.com/compute/cuda/12.3.2/local_installers/cuda_12.3.2_545.23.08_linux.run
sudo sh cuda_12.3.2_545.23.08_linux.run
# set ACMB_PATH environment variable
cd ../../
echo "export ACMB_PATH=$PWD" >> ~/.bashrc
source ~/.bashrc
