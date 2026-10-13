# Standalone CPU-only build of the post-processing stages and post-replay, without ggml or a GPU.
# For profiling and determinism tests of the mesh pipeline. Not part of the main build.
#   cmake -S tools/arctic-post -B build-post -G Ninja -DCMAKE_BUILD_TYPE=Release && ninja -C build-post
cmake_minimum_required(VERSION 3.16)
project(arctic_post CXX)
set(CMAKE_CXX_STANDARD 17)
get_filename_component(R "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
find_package(Threads REQUIRED)
set(SRCS mesh_glb tri_bvh mesh_shape quad_refinement quad_refinement_gates quad_refinement_io quad_components
         quad_topology quad_component_cover uv_overlap_cover uv_bake remesh_dc decimate_qem post_parallel post_mesh dual_grid)
set(SL "")
foreach(s ${SRCS})
  list(APPEND SL ${R}/src/${s}.cpp)
endforeach()
add_library(arctic_post_core STATIC ${SL} ${R}/thirdparty/xatlas/xatlas.cpp ${R}/thirdparty/meshoptimizer/simplifier.cpp)
target_include_directories(arctic_post_core PUBLIC ${R}/include ${R}/thirdparty/stb ${R}/thirdparty/xatlas
                           ${R}/thirdparty/meshoptimizer ${R}/thirdparty/fqms)
target_link_libraries(arctic_post_core PUBLIC Threads::Threads)
add_executable(post-replay ${R}/src/post_replay.cpp)
target_link_libraries(post-replay PRIVATE arctic_post_core)
add_executable(edgebench ${CMAKE_CURRENT_LIST_DIR}/edgebench.cpp)
target_link_libraries(edgebench PRIVATE Threads::Threads)
add_executable(post-equiv-test ${CMAKE_CURRENT_LIST_DIR}/post_equiv_test.cpp)
target_link_libraries(post-equiv-test PRIVATE arctic_post_core)
add_executable(dualgrid-bench ${CMAKE_CURRENT_LIST_DIR}/dualgrid_bench.cpp)
target_link_libraries(dualgrid-bench PRIVATE arctic_post_core)

# phase_lock.h (cross-process lock for the post GPU's QEM / bake phases): a header-only test, and decimate_qem.cpp compiled WITH
# TRELLIS_HAVE_GPU_DECIMATE against a stub GPU function so the real call site runs without CUDA. Driven by test_phase_lock.py.
add_executable(phase-lock-test ${CMAKE_CURRENT_LIST_DIR}/phase_lock_test.cpp)
target_include_directories(phase-lock-test PRIVATE ${R}/include)
target_link_libraries(phase-lock-test PRIVATE Threads::Threads)
add_executable(qem-lock-test ${CMAKE_CURRENT_LIST_DIR}/qem_lock_test.cpp ${R}/src/decimate_qem.cpp)
target_compile_definitions(qem-lock-test PRIVATE TRELLIS_HAVE_GPU_DECIMATE)
target_link_libraries(qem-lock-test PRIVATE arctic_post_core)
