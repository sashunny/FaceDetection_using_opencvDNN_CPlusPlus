
#include "FaceDetector.h"

faceDetector::faceDetector()
{

}


Mat faceDetector::Detectfaces(Mat &frame, const string caffeConfigFile, const string caffeWeightFile)
{

	float confidence_threshold = 0.5f;
    float nms_threshold	= 0.45f;
    Net net = readNetFromCaffe(caffeConfigFile, caffeWeightFile);
    vector<Rect> boxes;
    vector<float> confvect;

	Mat inputBlob = blobFromImage(frame, 1.0, Size(300, 300), Scalar(127.5,127.5,127.5), false, false);//104, 117, 123/127.5,127.5,127.5

    net.setInput(inputBlob, "data");
    Mat detection = net.forward("detection_out");
        

    Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
        
    for(int i = 0; i < detectionMat.rows; i++)
        {

        	float confidence = detectionMat.at<float>(i, 2);

        	if(confidence > confidence_threshold)
            {

                int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
                int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
                int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
                int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);
                int width = x2- x1;
                int height = y2 - y1;

                boxes.push_back((Rect(x1, y1, width,height)));
                confvect.push_back(confidence);
                cout<<"confidence :"<<confidence<<endl;

            }
        }

        vector<int> indices;
        NMSBoxes(boxes, confvect, confidence_threshold, nms_threshold, indices);

        Rect roi;

        for (size_t i = 0; i < indices.size(); ++i)
        {
            int idx = indices[i];
            Rect box = boxes[idx];
            rectangle(frame, cv::Point(box.x, box.y), cv::Point(box.x + box.width,box.y + box.height), Scalar(0, 255, 0), 3);
            namedWindow("FACE DETECTION USING DNN", WINDOW_NORMAL);
            imshow("FACE DETECTION USING DNN", frame);
            
            roi.x = box.x;
            roi.y = box.y;
            roi.width = box.width;
            roi.height = box.height;
        }
        waitKey(1);

        return frame(roi);
}


void faceDetector::createDataSet(Mat &img, const string &path, int &cnt, int &userID)
{
    string folderPath = path + "/" + to_string(userID);
    char* folderPath_char = const_cast<char*>(folderPath.c_str());

    mkdir(folderPath_char, 0777);
    string name = folderPath + "/outputIMG_" + to_string(cnt) + ".png";
    imwrite(name, img);
}

