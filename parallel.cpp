#include <opencv2/dnn/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>

using namespace cv;
using namespace dnn;
using namespace std;

// Globalne kolejki i zmienne do synchronizacji
queue<Mat> frameQueue;
queue<pair<Mat, vector<Mat>>> outputQueue;
mutex mtx;
condition_variable cv_frameQueue;
condition_variable cv_outputQueue;
bool finished = false;

void processFrame(Net net, vector<String> outputLayers) {
    while (true) {
        Mat frame;
        {
            unique_lock<mutex> lock(mtx);
            cv_frameQueue.wait(lock, [] { return !frameQueue.empty() || finished; });

            if (finished && frameQueue.empty())
                break;

            frame = frameQueue.front();
            frameQueue.pop();
        }

        Mat blob = blobFromImage(frame, 0.00392, Size(416, 416), Scalar(0, 0, 0), true, false);
        net.setInput(blob);
        vector<Mat> outs;
        net.forward(outs, outputLayers);

        {
            unique_lock<mutex> lock(mtx);
            outputQueue.push(make_pair(frame, outs));
            cv_outputQueue.notify_one();
        }
    }
}

void displayFrame() {
    while (true) {
        pair<Mat, vector<Mat>> result;
        {
            unique_lock<mutex> lock(mtx);
            cv_outputQueue.wait(lock, [] { return !outputQueue.empty() || finished; });

            if (finished && outputQueue.empty())
                break;

            result = outputQueue.front();
            outputQueue.pop();
        }

        Mat frame = result.first;
        vector<Mat> outs = result.second;

        for (auto& out : outs) {
            for (int i = 0; i < out.rows; ++i) {
                float* data = (float*)out.ptr<float>(i);
                float confidence = data[4];
                if (confidence > 0.5) {
                    int centerX = (int)(data[0] * frame.cols);
                    int centerY = (int)(data[1] * frame.rows);
                    int width = (int)(data[2] * frame.cols);
                    int height = (int)(data[3] * frame.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;
                    rectangle(frame, Point(left, top), Point(left + width, top + height), Scalar(0, 255, 0), 3);
                }
            }
        }

        imshow("Frame", frame);
        if (waitKey(1) == 'q')
            break;
    }
}

int main() {
    // Ładowanie YOLO
    String modelConfiguration = "../net/yolov3.cfg";
    String modelWeights = "../net/yolov3.weights";
    Net net = readNetFromDarknet(modelConfiguration, modelWeights);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    // Pobieranie nazw warstw
    vector<String> layerNames = net.getLayerNames();
    vector<String> outputLayers;
    for (auto& i : net.getUnconnectedOutLayers())
        outputLayers.push_back(layerNames[i - 1]);

    // Otwarcie kamery
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error: Unable to open the camera" << endl;
        return -1;
    }

    // Tworzenie wątków
    vector<thread> workers;
    int numThreads = 4;
    for (int i = 0; i < numThreads; ++i)
        workers.push_back(thread(processFrame, net, outputLayers));

    thread displayThread(displayFrame);

    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) {
            cerr << "Error: Frame capture failed or end of stream." << endl;
            break;
        }

        {
            unique_lock<mutex> lock(mtx);
            frameQueue.push(frame);
            cv_frameQueue.notify_one();
        }
    }

    // Ustawienie flagi zakończenia i notyfikacja wątków
    {
        unique_lock<mutex> lock(mtx);
        finished = true;
    }
    cv_frameQueue.notify_all();
    cv_outputQueue.notify_all();

    // Dołączenie wątków
    for (auto& t : workers)
        t.join();
    displayThread.join();

    cap.release();
    destroyAllWindows();

    return 0;
}

