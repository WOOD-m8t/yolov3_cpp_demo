// object dectertion uses the following model YOLOv3-416
// 
// @article{ yolov3,
//  title = {YOLOv3: An Incremental Improvement},
//  author = {Redmon, Joseph and Farhadi, Ali},
//  journal = {arXiv},
//  year = {2018}
//}
// title: Yolo test on Cuda enabled hardware
#include<iostream>
#include<fstream>
#include<sstream>

// Required for dnn modules.
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace dnn;

// confidence threshold
float conf_threshold = 0.5, nms = 0.4;

// image size for model input
int width = 416, height = 416;
// Object names 
vector<string> classes;

// function protoypes
void Clean_and_Draw(Mat& frame, const vector<Mat>& out);
void draw_box(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);
vector<String> getOutputsNames(const Net& net);

// Main function
int main(int argc, char** argv) {

    // load class nanes
    string classesFile = "coco.names";
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line)) classes.push_back(line);

    // load model weights and architecture
    String configuration = "yolov3.cfg";
    String model = "yolov3.weights";

    // Load the network
    Net net = readNetFromDarknet(configuration, model);
    Mat frame, blob;
    // test back end and set best set up availiable generally default is CPU
    auto a  = getAvailableBackends();
    if (!a.empty())
    {
         net.setPreferableBackend(a[a.size() - 1].first);
         net.setPreferableTarget(a[a.size() - 1].second);
    }
    // set up video input
    VideoCapture cap(0);
    // set loop flags
    int key = 1,quit = 1;

    while (quit != -1)
    {
        double start = getTickCount();
        cap >> frame;
        // convert image to blob NOTE::  scale factor & image size & mean subtraction are importent here. 
        blobFromImage(frame, blob,1/255.0, cv::Size(width, height), Scalar(0, 0, 0), true, false);
        // pass blob to neural network
        net.setInput(blob);
        // create array of martrix's each matrix is an output layer and the corrosponding data
        vector<Mat> outs;
        net.forward(outs, getOutputsNames(net));
        // post process the net output
        // Remove overlapping boxes and draws detections above the threshold on the frame
        Clean_and_Draw(frame, outs);
        // time calculations 
        double end = getTickCount();
        float  FPS = 1/ ((end - start) / getTickFrequency());
        string label = format("FPS: %.2f", FPS);
        putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 50,0));
        // Display image 
        imshow("Detected objects", frame);
        key = waitKey(1);
        // test for user input to quit program
        if (key == 'q' || key =='Q' || key =='27')
        {
            quit = -1;
       }
    }
    // clean up windows releases camera
    cap.release();
    cv::destroyAllWindows();
}

// Get the names of the output layers
vector<String> getOutputsNames(const Net& net)
{
    static vector<String> names;
    if (names.empty())
    {
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        vector<int> outLayers = net.getUnconnectedOutLayers();

        //get the names of all the layers in the network
        vector<String> layersNames = net.getLayerNames();

        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
            names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}

void Clean_and_Draw(Mat& frame, const vector<Mat>& outs)
{
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;
    // loop check each model output layer 
    for (size_t i = 0; i < outs.size(); ++i)
    {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > conf_threshold)
            {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }

    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms, indices);
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        Rect box = boxes[idx];
        draw_box(classIds[idx], confidences[idx], box.x, box.y,box.x + box.width, box.y + box.height, frame);
    }
}

// Draw the predicted bounding box
void draw_box(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
    //Draw a rectangle displaying the bounding box
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(200, 200, 50), 3);

    //Get the label for the class name and its confidence
    string label = format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ":" + label;
    }
    //Display the label at the top of the bounding box
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - round(1.5 * labelSize.height)), Point(left + round(1.5 * labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 1);
}

