rm -r release
mkdir release
g++ ./main.cpp -std=c++11 -fPIC -shared -o ./release/lib.so -pthread -O3 -march=native
