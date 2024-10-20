glslc shader.vert -o shader.vert.spv
glslc shader.frag -o shader.frag.spv
glslc raygen.rgen -o raygen.rgen.spv --target-env=vulkan1.2
glslc closesthit.rchit -o closesthit.rchit.spv --target-env=vulkan1.2
glslc miss.rmiss -o miss.rmiss.spv --target-env=vulkan1.2
rmdir /s /q build
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
cd Debug
app.exe
cd .../