#pragma once
#ifndef FACE_CONST_H
#define FACE_CONST_H

const float FACE_DET_THRESHOLD = 0.7f;
const int FACE_SEG_SIZE = 50;
const float FACE_SEG_THRESHOLD = 0.1f;
//const int FACE_SEG_SIZE = 100;
const int FACE_ALIGNED_SIZE = 100;
const int FACE_RECOGNITION_SIZE = 100;

//const int FACE_FEATURE_DIM_SIZE = 256;
const int FACE_FEATURE_DIM_SIZE = 512;

const float DEFAULT_FACE_RECOGNITION_THRESHOLD = 0.4f; 

const int FACE_NONE_NAME_THRESHOLD = 3;

const int VOTE_MIN_NUMBER = 5;
const int VOTE_MISS_NUMBER = 5;
const int HISTORY_VOTE_NUMBER = 15;

const int FACE_REG_WIDTH = 1280;
const int FACE_REG_HEIGHT = 720;
const float FACE_SCALE_STEP = 0.9f;
const int  POSE_NUN=3;
const float POSE_THRESHOLDS[3]={80.0f,50.0f,30.0f};

#endif
