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

add_executable(tests
    ${PROJECT_SOURCE_DIR}/main.cpp
    ${PROJECT_SOURCE_DIR}/matmul.cpp
)

target_link_libraries(tests
    openblas
    blis
)
