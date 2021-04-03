cmake_minimum_required(VERSION 3.4.1)
project(NCNN_SAMPLE)
include_directories("${CMAKE_SOURCE_DIR}/include")
set(CMAKE_INSTALL_RPATH "./build")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS
    "${CMAKE_EXE_LINKER_FLAGS}")

# Using Custom OpenCv pathC
# set(OpenCV_DIR "${CMAKE_SOURCE_DIR}/../sharedlibs/opencv/share/OpenCV")
set(ncnn_DIR /home/vuthede/ncnn/build/install/lib/cmake/ncnn/)
set(ncnn_INCLUDE_DIRS /home/vuthede/ncnn/build/install/include/ncnn/)
set(ncnn_LIBS /home/vuthede/ncnn/build/install/lib/libncnn.a)

find_package(OpenCV REQUIRED)
find_package(ncnn REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS
    "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")



# OpenCV and demo headers 
message("Opencv include: " ${OpenCV_INCLUDE_DIRS} )
message("ncnn include: " ${ncnn_INCLUDE_DIRS} )
message("Opencv libs: " ${OpenCV_LIBS} )
message("ncnn libs: " ${ncnn_LIBS} )


include_directories(${OpenCV_INCLUDE_DIRS} ${ncnn_INCLUDE_DIRS})

add_executable(benchmark benchmark_ncnn.cpp )
target_link_libraries(benchmark  ${OpenCV_LIBS} ${ncnn_LIBS})