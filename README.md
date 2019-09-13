# NetRAX
Phylogenetic Network Inference without ILS

1. **Install the dependencies.** On Ubuntu (and other Debian-based systems), you can simply run:
```
sudo apt-get install flex bison libgmp3-dev
```
For other systems, please make sure you have following packages/libraries installed:  
[`GNU Bison`](http://www.gnu.org/software/bison/) [`Flex`](http://flex.sourceforge.net/) [`GMP`](https://gmplib.org/)


2. **Build instructions**
(Tested on Ubuntu with GCC)
```
git clone --recurse-submodules https://github.com/lutteropp/NetRAX.git
cd NetRAX
mkdir build
cd build
cmake ..
make
```
