#g++ facedetection-demo.cpp -o facedetect -I/usr/include -L/usr/lib/aarch64-linux-gnu -lfacedetection -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_video

#include "FaceLandmark.h"
g++ facedetection-camera-landmark.cpp FaceLandmark.cpp Function.cpp \
-o facedetect-landmark \
-DCPU_ONLY \
-I/usr/include \
-L/usr/lib/aarch64-linux-gnu \
-L/home/wjq/caffe/build/lib \
-lfacedetection \
-lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_video \
-lprotobuf \
-lglog \
-lboost_thread -lboost_system \
-lcaffe \
-std=c++11
