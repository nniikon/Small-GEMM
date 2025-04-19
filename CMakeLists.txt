cmake_minimum_required(VERSION 3.28)
project(MatrixMul)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} \
    -O3 \
    -ffast-math \
    -DNDEBUG \
    -march=native"
)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} \
    -g \
    -O0 \
    -march=native \
    -Wall \
    -Wmissing-declarations \
    -Wcast-align \
    -Wcast-qual \
    -Wchar-subscripts \
    -Wconversion \
    -Wempty-body \
    -Wfloat-equal \
    -Wformat-nonliteral \
    -Wformat-security \
    -Wformat=2 \
    -Winline \
    -Wpacked \
    -Wpointer-arith \
    -Winit-self \
    -Wredundant-decls \
    -Wshadow \
    -Wstrict-overflow=2 \
    -Wswitch-default \
    -Wswitch-enum \
    -Wundef \
    -Wunreachable-code \
    -Wunused \
    -Wvariadic-macros \
    -Wno-nonnull \
    -Wno-missing-field-initializers \
    -Wno-narrowing \
    -Wno-varargs \
    -Wstack-protector \
    -fcheck-new \
    -fstack-protector \
    -fstrict-overflow \
    -fno-omit-frame-pointer -fPIC\
    -fsanitize=address,bool,bounds,enum,float-cast-overflow,float-divide-by-zero,integer-divide-by-zero,leak,nonnull-attribute,null,object-size,return,returns-nonnull-attribute,shift,signed-integer-overflow,undefined,unreachable,vla-bound,vptr"
)

set(BUILD_TYPE "" CACHE STRING "Build type: STATIC, SHARED, or EXECUTABLE")
set_property(CACHE BUILD_TYPE PROPERTY STRINGS STATIC SHARED EXECUTABLE)

if(NOT BUILD_TYPE)
    message(FATAL_ERROR "You must specify BUILD_TYPE (STATIC, SHARED, or EXECUTABLE)")
endif()

set(MATRIXMUL_SOURCES ${PROJECT_SOURCE_DIR}/matmul.cpp)

if(BUILD_STATIC)
    add_library(matrixmul STATIC ${MATRIXMUL_SOURCES})
endif()

if(BUILD_SHARED)
    add_library(matrixmul SHARED ${MATRIXMUL_SOURCES})
endif()

if(BUILD_TEST)
    add_executable(tests
        ${PROJECT_SOURCE_DIR}/test.cpp
        ${PROJECT_SOURCE_DIR}/matmul.cpp
    )

    target_link_libraries(tests PRIVATE openblas)
endif()

if(BUILD_TYPE STREQUAL "STATIC")
    add_library(matrixmul STATIC ${MATRIXMUL_SOURCES})

elseif(BUILD_TYPE STREQUAL "SHARED")
    add_library(matrixmul SHARED ${MATRIXMUL_SOURCES})

elseif(BUILD_TYPE STREQUAL "EXECUTABLE")
    add_executable(tests
        ${PROJECT_SOURCE_DIR}/test.cpp
        ${PROJECT_SOURCE_DIR}/matmul.cpp
    )

    target_link_libraries(tests PRIVATE openblas)

else()
    message(FATAL_ERROR "Invalid BUILD_TYPE: ${BUILD_TYPE}. Valid options: STATIC, SHARED, EXECUTABLE.")
endif()

