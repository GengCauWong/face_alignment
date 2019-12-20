#include "Function.h"
#include <numeric>

int get_blob_index(boost::shared_ptr< Net<float> > & net, char *query_blob_name)
{
	std::string str_query(query_blob_name);
	vector< string > const & blob_names = net->blob_names();
	for (unsigned int i = 0; i != blob_names.size(); ++i)
	{
		if (str_query == blob_names[i])
		{
			return i;
		}
	}
	LOG(FATAL) << "Unknown blob name: " << str_query;

	return -1;
}

int get_layer_index(boost::shared_ptr< Net<float> > & net, char *query_layer_name)
{
	std::string str_query(query_layer_name);
	vector< string > const & layer_names = net->layer_names();
	for (unsigned int i = 0; i != layer_names.size(); ++i)
	{
		if (str_query == layer_names[i])
		{
			return i;
		}
	}
	LOG(FATAL) << "Unknown layer name: " << str_query;

	return -1;
}

void get_blob_features_old(boost::shared_ptr< Net<float> > & net, float *data_ptr, char* layer_name)
{
	unsigned int id = get_layer_index(net, layer_name);
	const vector<Blob<float>*>& output_blobs = net->top_vecs()[id];
	for (unsigned int i = 0; i < output_blobs.size(); ++i)
	{
		switch (Caffe::mode()) {
		case Caffe::CPU:
			memcpy(data_ptr, output_blobs[i]->cpu_data(),
				sizeof(float) * output_blobs[i]->count());
			break;
#if 0
		case Caffe::GPU:
			cudaMemcpy(data_ptr, output_blobs[i]->gpu_data(),
				sizeof(float) * output_blobs[i]->count(), cudaMemcpyDeviceToHost);
			break;
#endif
		default:
			LOG(FATAL) << "Unknown Caffe mode.";
		}
	}
}

void get_blob_features(boost::shared_ptr< Net<float> > & net, float *&data_ptr, char* layer_name)
{
	unsigned int id = get_layer_index(net, layer_name);
	const vector<Blob<float>*>& output_blobs = net->top_vecs()[id];
	int size = output_blobs[0]->count();

	data_ptr = new float[output_blobs[0]->count()];
	for (unsigned int i = 0; i < 1; ++i)
	{
		switch (Caffe::mode()) {
		case Caffe::CPU:
			memcpy(data_ptr, output_blobs[i]->cpu_data(),
				sizeof(float) * output_blobs[i]->count());
			break;
#if 0
		case Caffe::GPU:
			cudaMemcpy(data_ptr, output_blobs[i]->gpu_data(),
				sizeof(float) * output_blobs[i]->count(), cudaMemcpyDeviceToHost);
			break;
#endif
		default:
			LOG(FATAL) << "Unknown Caffe mode.";
		}
	}
}

bool ReadMatToSegBlob(cv::Mat & cv_img, Blob<float>& dst, float B_mean, float G_mean, float R_mean){
	CHECK_EQ(3, cv_img.channels());
	if (3 != cv_img.channels())
	{
		return false;
	}

	dst.Reshape(1, cv_img.channels(), cv_img.rows, cv_img.cols);

	float* dst_data = dst.mutable_cpu_data();
	
	for (int c = 0; c < cv_img.channels(); ++c) {
		for (int h = 0; h < cv_img.rows; ++h) {
			for (int w = 0; w < cv_img.cols; ++w) {
				float temp = static_cast<float>(cv_img.at<cv::Vec3b>(h, w)[c]);
				if (c == 0)
				{
					*(dst_data++) = static_cast<float>(cv_img.at<cv::Vec3b>(h, w)[c]) - B_mean;
				}
				if (c == 1)
				{
					*(dst_data++) = static_cast<float>(cv_img.at<cv::Vec3b>(h, w)[c]) - G_mean;
				}
				if (c == 2)
				{
					*(dst_data++) = static_cast<float>(cv_img.at<cv::Vec3b>(h, w)[c]) - R_mean;
				}
			}
		}
	}

	return true;
}

bool ReadMatToASSDBlob(cv::Mat & cv_img, Blob<float>& dst){
	CHECK_EQ(3, cv_img.channels());
	if (3 != cv_img.channels())
	{
		return false;
	}

	dst.Reshape(1, cv_img.channels(), cv_img.rows, cv_img.cols);

	float* dst_data = dst.mutable_cpu_data();

	for (int c = 0; c < cv_img.channels(); ++c) {
		for (int h = 0; h < cv_img.rows; ++h) {
			for (int w = 0; w < cv_img.cols; ++w) {
				*(dst_data++) = (static_cast<float>(cv_img.at<cv::Vec3b>(h, w)[c]) - 127.5) / 127.5;
			}
		}
	}
	return true;
}

/*
bool ReadMatToASSDBlob(cv::Mat & cv_img, Blob<float>& dst){
	CHECK_EQ(3, cv_img.channels());
	if (3 != cv_img.channels())
	{
		return false;
	}

	dst.Reshape(1, cv_img.channels(), cv_img.rows, cv_img.cols);

	float* dst_data = dst.mutable_cpu_data();

	for (int c = 0; c < cv_img.channels(); ++c) {
		for (int h = 0; h < cv_img.rows; ++h) {
			for (int w = 0; w < cv_img.cols; ++w) {
				float temp = static_cast<float>(cv_img.at<cv::Vec3b>(h, w)[c]);
				if (c == 0)
				{
					*(dst_data++) = (static_cast<float>(cv_img.at<cv::Vec3b>(h, w)[c]) - 127.5) / 127.5;
				}
				if (c == 1)
				{
					*(dst_data++) = (static_cast<float>(cv_img.at<cv::Vec3b>(h, w)[c]) - 127.5) / 127.5;
				}
				if (c == 2)
				{
					*(dst_data++) = (static_cast<float>(cv_img.at<cv::Vec3b>(h, w)[c]) - 127.5) / 127.5;
				}
			}
		}
	}

	return true;
}
*/
bool ReadMatToDaixuBlob(cv::Mat & cv_img, Blob<float>& dst){
	CHECK_EQ(1, cv_img.channels());
	if (1 != cv_img.channels())
	{
		return false;
	}

	dst.Reshape(1, cv_img.channels(), cv_img.rows, cv_img.cols);

	float* dst_data = dst.mutable_cpu_data();

	for (int c = 0; c < cv_img.channels(); ++c) {
		for (int h = 0; h < cv_img.rows; ++h) {
			for (int w = 0; w < cv_img.cols; ++w) {
				*(dst_data++) = (static_cast<float>(cv_img.at<uchar>(h, w)) - 127.5) / 127.5;
			}
		}
	}
	return true;
}

bool ReadMatToDaixuBlob_back(cv::Mat & cv_img, Blob<float>& dst){
	CHECK_EQ(1, cv_img.channels());
	if (1 != cv_img.channels())
	{
		return false;
	}

	dst.Reshape(1, 1, 100, 100);

	float* dst_data = dst.mutable_cpu_data();

	for (int h = 0; h < 100; ++h) {
		for (int w = 0; w < 100; ++w) {
			*(dst_data++) = static_cast<float>(0.0);
		}
	}

	return true;
}

bool ReadMatToBlob(cv::Mat & cv_img, Blob<float>& dst){
	CHECK_EQ(3, cv_img.channels());
	if (3 != cv_img.channels())
	{
		return false;
	}

	dst.Reshape(1, cv_img.channels(), cv_img.rows, cv_img.cols);

	float* dst_data = dst.mutable_cpu_data();

	for (int c = 0; c < cv_img.channels(); ++c) {
		for (int h = 0; h < cv_img.rows; ++h) {
			for (int w = 0; w < cv_img.cols; ++w) {
				*(dst_data++) = static_cast<float>(cv_img.at<cv::Vec3b>(h, w)[c]);
			}
		}
	}

	return true;
}
/*
bool ReadSegBlobToMat_Float(Blob<float>& dst, cv::Mat &res_image){
	CHECK_EQ(dst.shape(2), res_image.rows);
	CHECK_EQ(dst.shape(3), res_image.cols);
	if (dst.shape(2) != res_image.rows)
	{
		return false;
	}
	if (dst.shape(3) != res_image.cols)
	{
		return false;
	}

	float* dst_data = dst.mutable_cpu_data();

	for (int i = 0; i < res_image.rows; i++)
	{
		float *p = res_image.ptr<float>(i);
		for (int j = 0; j < res_image.cols; j++)
		{
			p[j] = static_cast<float>(dst_data[dst.offset(0, 0, i, j)]);
		}
	}

	return true;
}*/

bool ReadPartSegBlobToMat_Float(Blob<float>& dst, vector <cv::Mat> &seg_part_images)
{
	float* dst_data = dst.mutable_cpu_data();
	for (int i = 0; i < 5; i++)
	{
		cv::Mat temp_part_seg = cv::Mat(FACE_SEG_SIZE, FACE_SEG_SIZE, CV_32FC1, dst_data + i * FACE_SEG_SIZE * FACE_SEG_SIZE);
		
		cv::Mat res_image(FACE_SEG_SIZE, FACE_SEG_SIZE, CV_8UC1);
		ExtractSegMat(temp_part_seg, res_image, 0.1);
		seg_part_images.push_back(temp_part_seg);
	}
	return true;
}

bool ReadASSDBlobToMat_Float(Blob<float>& dst, cv::Mat& score_map, cv::Mat &reg_map_x1, cv::Mat &reg_map_y1, cv::Mat &reg_map_x2, cv::Mat &reg_map_y2)
{
	CHECK_EQ(dst.shape(2), score_map.rows);
	CHECK_EQ(dst.shape(3), score_map.cols);
	if (dst.shape(2) != score_map.rows)
	{
		return false;
	}
	if (dst.shape(3) != score_map.cols)
	{
		return false;
	}

	float* dst_data = dst.mutable_cpu_data();

	score_map =  cv::Mat (score_map.rows, score_map.cols, CV_32FC1, dst_data);
	reg_map_x1 = cv::Mat(score_map.rows, score_map.cols, CV_32FC1, dst_data + 1 * score_map.rows * score_map.cols);
	reg_map_y1 = cv::Mat(score_map.rows, score_map.cols, CV_32FC1, dst_data + 2 * score_map.rows * score_map.cols);
	reg_map_x2 = cv::Mat(score_map.rows, score_map.cols, CV_32FC1, dst_data + 3 * score_map.rows * score_map.cols);
	reg_map_y2 = cv::Mat(score_map.rows, score_map.cols, CV_32FC1, dst_data + 4 * score_map.rows * score_map.cols);
	return true;
}

bool ExtractSegMat(cv::Mat &ori_image, cv::Mat &res_image, float seg_threshold){

	CHECK_EQ(ori_image.rows, res_image.rows);
	CHECK_EQ(ori_image.cols, res_image.cols);
	if (ori_image.rows != res_image.rows)
	{
		return false;
	}
	if (ori_image.cols != res_image.cols)
	{
		return false;
	}

	for (int i = 0; i < res_image.rows; i++)
	{
		for (int j = 0; j < res_image.cols; j++)
		{
			float temp = ori_image.at<float>(i, j);
			//res_image.at<uchar>(i, j) = 100;

			if (temp > seg_threshold){
				res_image.at<uchar>(i, j) = 255;
			}
			else{
				res_image.at<uchar>(i, j) = 0;
			}
		}
	}

	return true;
}

/*
bool PrintImageData(cv::Mat &cv_image){
	using namespace std;

	for (int i = 0; i < cv_image.rows; i++)
	{
		for (int j = 0; j < cv_image.cols; j++)
		{
			cout << int(cv_image.at<uchar>(i, j));
		}
		cout << endl;
	}

	return true;
}

bool PrintBlobData(Blob<float>& dst){
	using namespace std;
	float* dst_data = dst.mutable_cpu_data();

	for (int i = 0; i < dst.shape(2); i++)
	{
		for (int j = 0; j < dst.shape(3); j++)
		{
			cout << static_cast<float>(dst_data[dst.offset(0, 0, i, j)]) << " ";
		}
		cout << endl;
	}

	return true;
}

void resize_nearest(cv::Mat & matSrc, cv::Mat & matDst, float scale_ratio){
	cv::Size size = matSrc.size();
	matDst = cv::Mat(cv::Size(size.width*scale_ratio, size.height*scale_ratio), matSrc.type(), cv::Scalar::all(0));
	double scale_x = 1.0 / scale_ratio;
	double scale_y = 1.0 / scale_ratio;

#pragma omp parallel for num_threads(2)
	//#pragma omp parallel for
	for (int i = 0; i < matDst.cols; ++i)
	{
		int sx = cvFloor(i * scale_x);
		sx = std::min(sx, matSrc.cols - 1);
		//sx = sx <= matSrc.cols - 1 ?  sx : matSrc.cols - 1;
		for (int j = 0; j < matDst.rows; ++j)
		{
			int sy = cvFloor(j * scale_y);
			sy = std::min(sy, matSrc.rows - 1);
			//sy = sy <= matSrc.rows - 1 ? sy : matSrc.rows - 1;
			matDst.at<cv::Vec3b>(j, i) = matSrc.at<cv::Vec3b>(sy, sx);
		}
	}
}

void resize_liner(cv::Mat & matSrc, cv::Mat & matDst, float scale_ratio){
	cv::Size size = matSrc.size();
	matDst = cv::Mat(cv::Size(size.width*scale_ratio, size.height*scale_ratio), matSrc.type(), cv::Scalar::all(0));
	double scale_x = 1.0 / scale_ratio;
	double scale_y = 1.0 / scale_ratio;

	uchar* dataDst = matDst.data;
	int stepDst = matDst.step;
	uchar* dataSrc = matSrc.data;
	int stepSrc = matSrc.step;
	int iWidthSrc = matSrc.cols;
	int iHiehgtSrc = matSrc.rows;

#pragma omp parallel for
	for (int j = 0; j < matDst.rows; ++j)
	{
		float fy = (float)((j + 0.5) * scale_y - 0.5);
		int sy = cvFloor(fy);
		fy -= sy;
		sy = std::min(sy, iHiehgtSrc - 2);
		sy = std::max(0, sy);

		short cbufy[2];
		cbufy[0] = cv::saturate_cast<short>((1.f - fy) * 2048);
		cbufy[1] = 2048 - cbufy[0];

		for (int i = 0; i < matDst.cols; ++i)
		{
			float fx = (float)((i + 0.5) * scale_x - 0.5);
			int sx = cvFloor(fx);
			fx -= sx;

			if (sx < 0) {
				fx = 0, sx = 0;
			}
			if (sx >= iWidthSrc - 1) {
				fx = 0, sx = iWidthSrc - 2;
			}

			short cbufx[2];
			cbufx[0] = cv::saturate_cast<short>((1.f - fx) * 2048);
			cbufx[1] = 2048 - cbufx[0];

			for (int k = 0; k < matSrc.channels(); ++k)
			{
				*(dataDst + j*stepDst + 3 * i + k) = (*(dataSrc + sy*stepSrc + 3 * sx + k) * cbufx[0] * cbufy[0] +
					*(dataSrc + (sy + 1)*stepSrc + 3 * sx + k) * cbufx[0] * cbufy[1] +
					*(dataSrc + sy*stepSrc + 3 * (sx + 1) + k) * cbufx[1] * cbufy[0] +
					*(dataSrc + (sy + 1)*stepSrc + 3 * (sx + 1) + k) * cbufx[1] * cbufy[1]) >> 22;
			}
		}
	}
}
*/

bool CalculatePointsMean(cv::Point& mean_point, vector<cv::Point>& points){
	cv::Point sum = std::accumulate(points.begin(), points.end(), mean_point);
	cv::Point mean_point_loc(sum.x / points.size(), sum.y / points.size());
	mean_point = mean_point_loc;
	return true;
}

// copied from FaceRecognizerImpl.cpp
int GetLargestAreaRect(vector<cv::Rect> & face_rects){
	assert(face_rects.size() > 0);
	if (face_rects.size() == 0)
	{
		return -1;
	}

	int max_area = 0;
	int max_index = -1;
	int area = 0;
	for (int i = 0; i < face_rects.size(); i++)
	{
		cv::Rect & rect = face_rects[i];

		area = rect.width * rect.height;
		if (area > max_area)
		{
			max_area = area;
			max_index = i;
		}
	}

	return max_index;
}

// copied from Jeff 
double IOU(const cv::Rect& r1, const cv::Rect& r2)
{
	int x1 = std::max(r1.x, r2.x);
	int y1 = std::max(r1.y, r2.y);
	int x2 = std::min(r1.x + r1.width, r2.x + r2.width);
	int y2 = std::min(r1.y + r1.height, r2.y + r2.height);
	int w = std::max(0, (x2 - x1 + 1));
	int h = std::max(0, (y2 - y1 + 1));
	double inter = w * h;
	double o = inter / (r1.area() + r2.area() - inter);
	return (o >= 0) ? o : 0;
}

// ref by http://blog.csdn.net/duinodu/article/details/61651390
void nms(vector<cv::Rect>& proposals, const double nms_threshold)
{
	vector<int> scores;
	for (auto i : proposals) scores.push_back(i.area());

	vector<int> index;
	for (int i = 0; i < scores.size(); ++i){
		index.push_back(i);
	}

	sort(index.begin(), index.end(), [&](int a, int b){
		return scores[a] > scores[b];
	});

	vector<bool> del(scores.size(), false);
	for (size_t i = 0; i < index.size(); i++){
		if (!del[index[i]]){
			for (size_t j = i + 1; j < index.size(); j++){
				if (IOU(proposals[index[i]], proposals[index[j]]) > nms_threshold){
					del[index[j]] = true;
				}
			}
		}
	}

	vector<cv::Rect> new_proposals;
	for (const auto i : index){
		if (!del[i]) new_proposals.push_back(proposals[i]);
	}
	proposals = new_proposals;
}
