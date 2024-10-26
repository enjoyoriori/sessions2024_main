glslc shader.vert -o shader.vert.spv
glslc shader.frag -o shader.frag.spv
glslc depthPrePass.frag -o depthPrePass.frag.spv
rmdir /s /q build
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
cd Debug
app.exe
cd .../