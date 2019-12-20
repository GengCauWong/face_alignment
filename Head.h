#pragma once
#include <iostream>
#include "caffe/caffe.hpp"
#include <vector>
#include <string>
#include <sys/io.h>
#include "Const.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace caffe;
//#define TRACE 1
// _DEBUG may conflict with other libs when not building dlls...
#ifndef _DEBUG_MODE
#define tcout 0 && cout
#else
#define tcout cout
#endif
