#pragma once
#include "Head.h"
using namespace caffe;


bool ReadMatToBlob(cv::Mat & cv_img, Blob<float>& dst);
int get_layer_index(boost::shared_ptr< Net<float> > & net, char *query_layer_name);
void get_blob_features(boost::shared_ptr< Net<float> > & net, float *&data_ptr, char* layer_name);
int get_blob_index(boost::shared_ptr< Net<float> > & net, char *query_blob_name);
bool ReadMatToSegBlob(cv::Mat & cv_img, Blob<float>& dst, float B_mean, float G_mean, float R_mean);
bool ExtractSegMat(cv::Mat &ori_image, cv::Mat &res_image, float seg_threshold);
//bool PrintBlobData(Blob<float>& dst);
//bool ReadSegBlobToMat_Float(Blob<float>& dst, cv::Mat &res_image);
//bool ReadSegBlobToMat_UChar(Blob<float>& dst, cv::Mat &res_image);
//bool PrintImageData(cv::Mat &cv_image);
//void resize_nearest(cv::Mat & matSrc, cv::Mat & matDst, float scale_ratio);
//void resize_liner(cv::Mat & matSrc, cv::Mat & matDst, float scale_ratio);

bool ReadMatToASSDBlob(cv::Mat & cv_img, Blob<float> & dst);
bool ReadASSDBlobToMat_Float(Blob<float>& dst, cv::Mat & score_map, cv::Mat & reg_map_x1, cv::Mat & reg_map_y1, cv::Mat & reg_map_x2, cv::Mat & reg_map_y2);
bool CalculatePointsMean(cv::Point & mean_points, vector<cv::Point> & points);
bool ReadPartSegBlobToMat_Float(Blob<float>& dst, vector <cv::Mat> &seg_part_images);
bool ReadMatToDaixuBlob(cv::Mat & cv_img, Blob<float> & dst);

int GetLargestAreaRect(vector<cv::Rect> & face_rects);
void nms(vector<cv::Rect>& proposals, const double nms_threshold);

template <typename T>
void ClearVector( vector< T >& vt )
{
    vector< T > vTemp;
    vTemp.swap( vt );
}
