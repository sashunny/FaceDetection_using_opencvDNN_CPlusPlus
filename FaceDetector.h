
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>

#include <bits/stdc++.h> 
#include <sys/stat.h> 
#include <sys/types.h> 

using namespace std;
using namespace cv;
using namespace cv::dnn;

class faceDetector
{
	public:
		faceDetector();
		Mat Detectfaces(Mat &frame, const string caffeConfigFile, const string caffeWeightFile);
		void createDataSet(Mat &img, const string &path, int &cnt, int &userID);


};
