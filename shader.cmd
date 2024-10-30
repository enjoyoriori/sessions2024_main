glslc shader.vert -o shader.vert.spv
glslc shader.geom -o shader.geom.spv
glslc shader.frag -o shader.frag.spv
glslc lighting_shader.frag -o lighting_shader.frag.spv
glslc lighting_shader.vert -o lighting_shader.vert.spv
rmdir /s /q build
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
cd Debug
app.exe
cd .../