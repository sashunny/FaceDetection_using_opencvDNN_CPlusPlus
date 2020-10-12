
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
	const string prototxt = "../model/deploy.prototxt";
	const string folderPath = "../OutputData";
	int cnt = 0, userID;
	cout<<"Enter user id :"<<endl;
	cin>>userID;

	char* OutputData_path = const_cast<char*>(folderPath.c_str());

    mkdir(OutputData_path, 0777);

    Mat frame;

    while(cap.grab())
    {
        
        cap.retrieve(frame);

		Mat cropped_img = fd.Detectfaces(frame, prototxt, weight);

		fd.createDataSet(cropped_img, folderPath, cnt, userID);

		cnt++;
	}
}


