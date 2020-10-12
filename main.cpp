
#include "FaceDetector.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

int main()
{
	faceDetector fd;

	VideoCapture cap("../video/sample.webm");
	const string weight = "../model/res10_300x300_ssd_iter_140000.caffemodel";
	const string  prototxt = "../model/deploy.prototxt";

    cv::Mat frame;

    while(cap.grab())
    {
        
        cap.retrieve(frame);

		fd.Detectfaces(frame, prototxt, weight);
	}
}


