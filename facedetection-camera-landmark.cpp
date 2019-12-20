#include <stdio.h>
#include <string.h>
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "facedetect-dll.h"
#include "FaceLandmark.h"
//#include "FaceLandmark.cpp"

//define the buffer size. Do not change the size!
#define DETECT_BUFFER_SIZE 0xC004
using namespace cv;
using namespace std;

//int main(int argc, char* argv[])
int main()
{
    cv::VideoCapture capture;
    capture.open(0);
    if (!capture.isOpened())
    {
        std::cout << "video capture failed..." << std::endl;
        return 0;
    }
    cv::Mat src;
    cv::Mat image;
    cv::namedWindow("video test", CV_WINDOW_NORMAL);
    while (true)
    {
        src.release();
        capture >> src;

    #if 1 //镜像处理
        Mat map_x;
	Mat map_y;
	map_x.create( src.size(), CV_32FC1);
	map_y.create( src.size(), CV_32FC1);
	for( int i = 0; i < src.rows; ++i)
	{
		for( int j = 0; j < src.cols; ++j)
		{
			map_x.at<float>(i, j) = (float) (src.cols - j) ;
			map_y.at<float>(i, j) = (float) i;//(src.rows - i) ;
		}
	}
	remap(src, image, map_x, map_y, CV_INTER_LINEAR);


    #endif

        cv::Mat gray;
        cv::cvtColor(image, gray, CV_BGR2GRAY); 
     
        int * pResults = NULL;
        //pBuffer is used in the detection functions.
        //If you call functions in multiple threads, please create one buffer for each thread!
        unsigned char * pBuffer = (unsigned char *)malloc(DETECT_BUFFER_SIZE);
        if (!pBuffer)
        {
            fprintf(stderr, "Can not alloc buffer.\n");
            return -1;
        }

  
    

	#if 0
        ///////////////////////////////////////////
        // frontal face detection / 68 landmark detection
        // it's fast, but cannot detect side view faces
        //////////////////////////////////////////
        //!!! The input image must be a gray one (single-channel)
        //!!! DO NOT RELEASE pResults !!!
        pResults = facedetect_frontal(pBuffer, (unsigned char*)(gray.ptr(0)), gray.cols, gray.rows, (int)gray.step,
            1.2f, 2, 48, 0);

        printf("%d faces detected.\n", (pResults ? *pResults : 0));
        Mat result_frontal = image.clone();
        //print the detection results
        for (int i = 0; i < (pResults ? *pResults : 0); i++)
        {
            short * p = ((short*)(pResults + 1)) + 6 * i;
            int x = p[0];
            int y = p[1];
            int w = p[2];
            int h = p[3];
            int neighbors = p[4];
            int angle = p[5];

            printf("face_rect=[%d, %d, %d, %d], neighbors=%d, angle=%d\n", x, y, w, h, neighbors, angle);
            rectangle(result_frontal, Rect(x, y, w, h), Scalar(0, 255, 0), 2);
        }
        imshow("video test", result_frontal);
	#endif

	#if 0
        ///////////////////////////////////////////
        // frontal face detection designed for video surveillance / 68 landmark detection
        // it can detect faces with bad illumination.
        //////////////////////////////////////////
        //!!! The input image must be a gray one (single-channel)
        //!!! DO NOT RELEASE pResults !!!
        pResults = facedetect_frontal_surveillance(pBuffer, (unsigned char*)(gray.ptr(0)), gray.cols, gray.rows, (int)gray.step,
            1.2f, 2, 48, 0);
        printf("%d faces detected.\n", (pResults ? *pResults : 0));
        Mat result_frontal_surveillance = image.clone();;
        //print the detection results
        for (int i = 0; i < (pResults ? *pResults : 0); i++)
        {
            short * p = ((short*)(pResults + 1)) + 142 * i;
            int x = p[0];
            int y = p[1];
            int w = p[2];
            int h = p[3];
            int neighbors = p[4];
            int angle = p[5];

            printf("face_rect=[%d, %d, %d, %d], neighbors=%d, angle=%d\n", x, y, w, h, neighbors, angle);
            rectangle(result_frontal_surveillance, Rect(x, y, w, h), Scalar(0, 255, 0), 2);
        }
        imshow("video test", result_frontal_surveillance);
	#endif

        ///////////////////////////////////////////
        // multiview face detection / 68 landmark detection
        // it can detect side view faces, but slower than facedetect_frontal().
        //////////////////////////////////////////
        //!!! The input image must be a gray one (single-channel)
        //!!! DO NOT RELEASE pResults !!!
        pResults = facedetect_multiview(pBuffer, (unsigned char*)(gray.ptr(0)), gray.cols, gray.rows, (int)gray.step,
            1.2f, 2, 48, 0);

        printf("%d faces detected.\n", (pResults ? *pResults : 0));
        Mat result_multiview = image.clone();;
        //print the detection results
	vector<cv::Rect> rects;

        for (int i = 0; i < (pResults ? *pResults : 0); i++)
        {
	    cv::Rect rect;
            short * p = ((short*)(pResults + 1)) + 6 * i;
            rect.x = p[0];
            rect.y = p[1];
            rect.width = p[2];
            rect.height = p[3];
	    rects.push_back(rect);
            int neighbors = p[4];
            int angle = p[5];

            printf("face_rect=[%d, %d, %d, %d], neighbors=%d, angle=%d\n", p[0], p[1], p[2], p[3], neighbors, angle);
            rectangle(result_multiview, Rect(p[0], p[1], p[2], p[3]), Scalar(0, 255, 0), 2);
        }
	imshow("video test", result_multiview);
	
	FaceLandmark flm=FaceLandmark("face_landmark.caffemodel", "face_landmark.prototxt", 0, 0);
	vector<cv::Mat> alignedImages;
	for (int i = 0; i < rects.size(); i++) {
	    vector<int> landmarks;
	    cv::Mat aligned_image;            
	    flm.GetLanrmarks(image, rects[i], landmarks);
	    flm.AlignFace(image,landmarks, aligned_image);
	    alignedImages.push_back(aligned_image);
	    string name = "./image/";
	    name = name + to_string(i) + ".jpg";
	    cv::imwrite(name, aligned_image);
	    imshow("video test2", aligned_image);
	}





        ///////////////////////////////////////////
        // reinforced multiview face detection / 68 landmark detection
        // it can detect side view faces, better but slower than facedetect_multiview().
        //////////////////////////////////////////
        //!!! The input image must be a gray one (single-channel)
        //!!! DO NOT RELEASE pResults !!!
	#if 0
        pResults = facedetect_multiview_reinforce(pBuffer, (unsigned char*)(gray.ptr(0)), gray.cols, gray.rows, (int)gray.step,
            1.2f, 3, 48, 0);

        printf("%d faces detected.\n", (pResults ? *pResults : 0));
        Mat result_multiview_reinforce = image.clone();;
        //print the detection results
        for (int i = 0; i < (pResults ? *pResults : 0); i++)
        {
            short * p = ((short*)(pResults + 1)) + 142 * i;
            int x = p[0];
            int y = p[1];
            int w = p[2];
            int h = p[3];
            int neighbors = p[4];
            int angle = p[5];

            printf("face_rect=[%d, %d, %d, %d], neighbors=%d, angle=%d\n", x, y, w, h, neighbors, angle);
            rectangle(result_multiview_reinforce, Rect(x, y, w, h), Scalar(0, 255, 0), 2);
        }
        imshow("video test", result_multiview_reinforce);
	#endif
        waitKey(100);

        //release the buffer
        free(pBuffer);

    }
    return 0;
}
