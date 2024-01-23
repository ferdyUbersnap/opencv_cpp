# Build

build using cmake development mode
```
cmake --build .\build\
.\build\Debug\OpenCVcpp.exe 
```

build using cmake production mode
```
cmake --build .\build\ --config Release
.\build\Release\OpenCVcpp.exe 
```

how to run
./your_program.exe_path image_path brightnes 10 highlight 1.5 white 0.8
example
```
.\build\Debug\OpenCVcpp.exe  image.jpg brighnes 10 exposure 50 white 10 black 10
```

how to build wasm
```
emcc ubersnap.cpp -I ./ -o ./wasm/output.html
```