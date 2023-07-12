cmake_minimum_required(VERSION 3.10)
project(FeatureMatcherDemo)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)

# ���ñ�������
set(CMAKE_BUILD_TYPE Release)

# Find OpenCV
set(OpenCV_INCLUDE_DIRS "./opencv/include")
set(OpenCV_LIBS "../opencv/lib/opencv_world455")

# Find ONNX Runtime
set(ONNXRUNTIME_INCLUDE_DIRS "./onnxruntime-win-x64-1.15.1/include")
set(ONNXRUNTIME_LIBRARIES "../onnxruntime-win-x64-1.15.1/lib/onnxruntime")

# ���ӿ�ִ���ļ�������Ĳ���·��
add_library(feature_matcher SHARED common.h superpoint.h superpoint.cpp  lightglue.h lightglue.cpp)
target_include_directories(feature_matcher PRIVATE ${OpenCV_INCLUDE_DIRS} ${ONNXRUNTIME_INCLUDE_DIRS})
target_link_libraries(feature_matcher ${OpenCV_LIBS} ${ONNXRUNTIME_LIBRARIES})
target_compile_definitions(feature_matcher PRIVATE FEATURE_MATCHER_EXPORTS)

# ��������cppDemo��ִ���ļ�
add_executable(cppDemo common.h cppDemo.cpp)
target_include_directories(cppDemo PRIVATE ${OpenCV_INCLUDE_DIRS} ${ONNXRUNTIME_INCLUDE_DIRS})
target_link_libraries(cppDemo
                      ${OpenCV_LIBS}
                      feature_matcher)
