list(APPEND CUDA_NVCC_FLAGS "-std=c++11;-DGOOGLE_CUDA=1;-Xcompiler")

find_library(TF_LIB_FILE NAMES tensorflow_framework PATHS ${TF_LIB} REQUIRED)

execute_process(
   COMMAND python -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_compile_flags()))"
   OUTPUT_VARIABLE TF_CFLAGS
)

execute_process(
   COMMAND python -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_link_flags()))"
   OUTPUT_VARIABLE TF_LFLAGS
)

message(STATUS ${TF_CFLAGS})
message(STATUS ${TF_LFLAGS})

# add conebeam ops
set(
   CONEBEAM_SRC
   ../torch-radon/src/forward.cu
   forward.cc
   ../torch-radon/src/backprojection.cu
   backprojection.cc
   ../torch-radon/src/parameter_classes.cu
   ../torch-radon/src/texture.cu
   ../torch-radon/src/log.cpp
)
cuda_add_library(conebeam ${CONEBEAM_SRC} SHARED)
target_include_directories(conebeam PRIVATE ${TF_INCLUDE})
target_link_libraries(conebeam ${TF_LIB_FILE})
target_compile_definitions(conebeam PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
