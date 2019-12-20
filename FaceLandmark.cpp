#include "FaceLandmark.h"
#include "caffe/caffe.hpp"
 using namespace std;
using namespace caffe;

const string out_layer = "sigmoid_predict_conv";

const float B_mean = 104.008f;
const float G_mean = 116.669f;
const float R_mean = 122.675f;

bool FaceLandmark::LoadModel(){
	if (m_gpu_mode)
	{
		Caffe::set_mode(Caffe::GPU);
		Caffe::SetDevice(m_gpu_id);
	}
	else{
		Caffe::set_mode(Caffe::CPU);
	}

	Phase phase = TEST;
	m_net.reset(new caffe::Net<float>(m_param_file, phase));
	m_net->CopyTrainedLayersFrom(m_deploy_file);

	return m_net != NULL;
}

bool FaceLandmark::GetLanrmarks(cv::Mat & image, cv::Rect & face_rect, vector<int> & landmarks,vector <float> &scores){
	double t_pre = (double)cvGetTickCount();
	cv::Mat face_region = image(face_rect).clone();

	cv::Size image_size = face_region.size();
	cv::Mat resized_image;
	cv::Mat image_pad(m_face_seg_size, m_face_seg_size, CV_8UC3, cv::Scalar(0)); // every point is zero value
	float resize_scale = float(m_face_seg_size) / std::max(image_size.height, image_size.width);
	resize(face_region, resized_image, cv::Size(0, 0), resize_scale, resize_scale);

	int top_pad = 0;
	int left_pad = 0;
	if (image_size.height > image_size.width)
	{
		top_pad = 0;
		left_pad = int((m_face_seg_size - resized_image.cols) / 2.0);
	}
	else
	{
		left_pad = 0;
		top_pad = int((m_face_seg_size - resized_image.rows) / 2.0);
	}
	cv::Rect temp_rect(left_pad, top_pad, resized_image.cols, resized_image.rows);
	resized_image.convertTo(image_pad(temp_rect), resized_image.type());


	int blob_id = get_blob_index(m_net, (char*)"data");
	if (blob_id < 0)
	{
		return false;
	}

	vector<Blob<float>* > dst_vec;
	boost::shared_ptr<Blob<float> > blob;
	blob = m_net->blobs()[blob_id];

	
	if (!ReadMatToSegBlob(image_pad, *blob, B_mean, G_mean, R_mean)){
		return false;
	}

	
	t_pre = (double)cv::getTickCount() - t_pre;
	//printf("pre process time = %g ms\n", t_pre *1000. / ((double)cv::getTickFrequency()));

	double t_forward = (double)cvGetTickCount();
	const vector<Blob<float>*>& res_blobs = m_net->ForwardPrefilled();
	t_forward = (double)cv::getTickCount() - t_forward;
	//printf("forward time = %g ms\n", t_forward *1000. / ((double)cv::getTickFrequency()));

	double t_post = (double)cvGetTickCount();
	Blob<float>* output_layer = res_blobs[0];	
	vector<cv::Mat> seg_part_images;

	if (!ReadPartSegBlobToMat_Float(*output_layer, seg_part_images)){
		return false;
	}
	//ConvertPartMapToLocation(seg_part_images, landmarks, seg_threshold);
	// I'd better post process here...
	//vector <int> temp_location = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	for (int i = 0; i < 5; i++)
	{
		int temp_x = 0;
		int temp_y = 0;
		cv::Scalar sum = cv::sum(seg_part_images[i]);
		//cout << "sum value " << sum.val[0] << endl;
		double minVal; double maxVal;
		if (sum.val[0] > 0)
		{

			cv::Point minLoc; cv::Point maxLoc;
			minMaxLoc(seg_part_images[i], &minVal, &maxVal, &minLoc, &maxLoc);
			if (maxVal > FACE_SEG_THRESHOLD){
				//cout << "there is value over threshold " << maxLoc.x << " " << maxLoc.y << endl;
				temp_x = maxLoc.x;
				temp_y = maxLoc.y;
			}
		}

		scores.push_back(float(maxVal));
		//convert to original coordinate...
		temp_x = temp_x - left_pad;
		temp_y = temp_y - top_pad;
		temp_x = int(round(temp_x / resize_scale));
		temp_y = int(round(temp_y / resize_scale));
		//cout << "landmark_x: " << temp_x << " landmark_y: " << temp_y << endl;
		// we have to convert the coordinates to the whole image to cope with daixu's alignment tools.
		
		landmarks.push_back(temp_x + face_rect.x);
		landmarks.push_back(temp_y + face_rect.y);
	}
	t_post = (double)cv::getTickCount() - t_post;
	//printf("post process time = %g ms\n", t_post *1000.  / ((double)cv::getTickFrequency()));

	return true;
}


bool FaceLandmark::GetLanrmarks(cv::Mat & image, cv::Rect & face_rect, vector<int> & landmarks){
	double t_pre = (double)cvGetTickCount();
	cv::Mat face_region = image(face_rect).clone();

	cv::Size image_size = face_region.size();
	cv::Mat resized_image;
	cv::Mat image_pad(m_face_seg_size, m_face_seg_size, CV_8UC3, cv::Scalar(0)); // every point is zero value
	float resize_scale = float(m_face_seg_size) / std::max(image_size.height, image_size.width);
	resize(face_region, resized_image, cv::Size(0, 0), resize_scale, resize_scale);

	int top_pad = 0;
	int left_pad = 0;
	if (image_size.height > image_size.width)
	{
		top_pad = 0;
		left_pad = int((m_face_seg_size - resized_image.cols) / 2.0);
	}
	else
	{
		left_pad = 0;
		top_pad = int((m_face_seg_size - resized_image.rows) / 2.0);
	}
	cv::Rect temp_rect(left_pad, top_pad, resized_image.cols, resized_image.rows);
	resized_image.convertTo(image_pad(temp_rect), resized_image.type());


	int blob_id = get_blob_index(m_net, (char*)"data");
	if (blob_id < 0)
	{
		return false;
	}

	vector<Blob<float>* > dst_vec;
	boost::shared_ptr<Blob<float> > blob;
	blob = m_net->blobs()[blob_id];


	if (!ReadMatToSegBlob(image_pad, *blob, B_mean, G_mean, R_mean)){
		return false;
	}


	t_pre = (double)cv::getTickCount() - t_pre;
	//printf("pre process time = %g ms\n", t_pre *1000. / ((double)cv::getTickFrequency()));

	double t_forward = (double)cvGetTickCount();
	const vector<Blob<float>*>& res_blobs = m_net->ForwardPrefilled();
	t_forward = (double)cv::getTickCount() - t_forward;
	//printf("forward time = %g ms\n", t_forward *1000. / ((double)cv::getTickFrequency()));

	double t_post = (double)cvGetTickCount();
	Blob<float>* output_layer = res_blobs[0];
	vector<cv::Mat> seg_part_images;

	if (!ReadPartSegBlobToMat_Float(*output_layer, seg_part_images)){
		return false;
	}
	//ConvertPartMapToLocation(seg_part_images, landmarks, seg_threshold);
	// I'd better post process here...
	//vector <int> temp_location = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	double minVal; double maxVal;
	for (int i = 0; i < 5; i++)
	{
		int temp_x = 0;
		int temp_y = 0;
		cv::Scalar sum = cv::sum(seg_part_images[i]);
		//cout << "sum value " << sum.val[0] << endl;

		if (sum.val[0] > 0)
		{
			cv::Point minLoc; cv::Point maxLoc;
			minMaxLoc(seg_part_images[i], &minVal, &maxVal, &minLoc, &maxLoc);
			if (maxVal > FACE_SEG_THRESHOLD){
				//cout << "there is value over threshold " << maxLoc.x << " " << maxLoc.y << endl;
				temp_x = maxLoc.x;
				temp_y = maxLoc.y;
			}
		}

		//convert to original coordinate...
		temp_x = temp_x - left_pad;
		temp_y = temp_y - top_pad;
		temp_x = int(round(temp_x / resize_scale));
		temp_y = int(round(temp_y / resize_scale));
		//cout << "landmark_x: " << temp_x << " landmark_y: " << temp_y << endl;
		// we have to convert the coordinates to the whole image to cope with daixu's alignment tools.

		landmarks.push_back(temp_x + face_rect.x);
		landmarks.push_back(temp_y + face_rect.y);
	}
	t_post = (double)cv::getTickCount() - t_post;
	//printf("post process time = %g ms\n", t_post *1000.  / ((double)cv::getTickFrequency()));

	return true;
}


bool FaceLandmark::AlignFace(cv::Mat & image, vector<int> & landmarks, cv::Mat & aligned_image){
	cv::Point2f dstTri[3];
	double scl_left = 0.45;
	double scl_right = 0.45;
	double scl_up = 0.70;
	double scl_down = 0.50;
	double height = FACE_ALIGNED_SIZE;
	double width = FACE_ALIGNED_SIZE;
	double aw = width / (scl_left + scl_right + 1.0);
	double ah = height / (scl_up + scl_down + 1.0);
	dstTri[0] = cv::Point2f(aw * scl_left, ah*scl_up);
	dstTri[1] = cv::Point2f(aw*(1 + scl_left), ah*scl_up);
	dstTri[2] = cv::Point2f(width / 2.0, ah*(1 + scl_up));

	cv::Point2f srcTri[3];
	//lefteye_x, lefteye_y, righteye_x, righteye_y, nose_x, nose_y,leftmouth_x,leftmouth_y, rightmouth_x, rightmouth_y
	srcTri[0] = cv::Point2f(landmarks[0], landmarks[1]);
	srcTri[1] = cv::Point2f(landmarks[2], landmarks[3]);
	srcTri[2] = cv::Point2f((landmarks[6] + landmarks[8]) / 2.0, (landmarks[7] + landmarks[9]) / 2.0);

	cv::Mat warp_mat(2, 3, CV_32FC1);
	warp_mat = getAffineTransform(srcTri, dstTri);
	cv::Mat warp_dst;
	warpAffine(image, warp_dst, warp_mat, cv::Size(FACE_ALIGNED_SIZE, FACE_ALIGNED_SIZE), cv::BORDER_CONSTANT);
	aligned_image = warp_dst;
	return true;
}
