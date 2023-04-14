cmake_minimum_required (VERSION 2.8)

# 项目信息
project (DigitalRecognize)

# 指定生成目标
add_executable(DigitalRecognize DigitalRecognize.cpp "NeuralNet.h" "NeuralNet.cpp")