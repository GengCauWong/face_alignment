#pragma once
#include "Head.h"
#include "Function.h"
#include <iostream>
//#include "FaceLandmark.cpp"
using namespace std;
using namespace caffe;

class FaceLandmark{
private:
	string m_deploy_file;
	bool m_gpu_mode;
	string m_param_file;
	int m_gpu_id;
	boost::shared_ptr<caffe::Net<float> > m_net;
	int m_face_seg_size;
public:
	FaceLandmark(string deploy_file, string param_file, bool gpu_mode = true, int gpu_id = 0)
		:m_deploy_file(deploy_file),
		m_param_file(param_file),
		m_gpu_mode(gpu_mode),
		m_gpu_id(gpu_id),
		//m_net(NULL),
		m_face_seg_size(FACE_SEG_SIZE)
	{
		if (m_deploy_file.c_str() == NULL) {
			cout << "Deploy file is not exist!" << endl;
			exit(0);
			return;
		}

		if (m_param_file.c_str()== NULL) {
			cout << "Parameter file is not exist!" << endl;
			exit(0);
			return;
		}
		LoadModel();
	}

	bool LoadModel();
	
	bool GetLanrmarks(cv::Mat & image, cv::Rect & face_rect, vector<int> & landmarks,vector <float> &scores);
	bool GetLanrmarks(cv::Mat & image, cv::Rect & face_rect, vector<int> & landmarks);

	bool AlignFace(cv::Mat & image, vector<int> & landmarks, cv::Mat & aligned_image);
};
