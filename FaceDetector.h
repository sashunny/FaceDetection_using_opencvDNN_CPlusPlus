
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>

using namespace std;
using namespace cv;
using namespace cv::dnn;

class faceDetector
{
	public:
		faceDetector();
		void Detectfaces(Mat &frame, const string caffeConfigFile, const string caffeWeightFile);
};
