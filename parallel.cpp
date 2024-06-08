
#include <atomic>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <queue>
#include <thread>
#include <unordered_map>
#include <vector>

using namespace std;
using namespace cv;
using namespace dnn;

mutex mtx;
condition_variable cvCapture, cvProcess;
queue<Mat> framesForProcessing;
queue<Mat> framesForDisplay;
atomic<bool> processing(true);
atomic<bool> capturing(true);
mutex objMtx; // Mutex for objects map

void drawPred(int id, int left, int top, int right, int bottom, Mat &frame) {
  rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 178, 50),
            3);

  string label = format("ID %d", id);
  int baseLine;
  Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
  top = max(top, labelSize.height);
  rectangle(frame, Point(left, top - round(1.5 * labelSize.height)),
            Point(left + round(1.5 * labelSize.width), top + baseLine),
            Scalar(255, 255, 255), FILLED);
  putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75,
          Scalar(0, 0, 0), 1);
}

void captureFrames(VideoCapture &cap) {
  Mat frame;
  while (capturing) {
    cap >> frame;
    if (frame.empty()) {
      cout << "Empty frame detected, stopping capture." << endl;
      break;
    }
    {
      lock_guard<mutex> lock(mtx);
      framesForProcessing.push(frame.clone());
    }
    cvCapture.notify_one();
    this_thread::sleep_for(chrono::milliseconds(30));
  }
  capturing = false;
  cvCapture.notify_all();
}

void processFrame(Net &net, const vector<string> &outNames,
                  unordered_map<int, Point2f> &objects, int &nextObjectId) {
  Mat frame;
  {
    unique_lock<mutex> lock(mtx);
    cvCapture.wait(lock,
                   [] { return !framesForProcessing.empty() || !capturing; });
    if (!framesForProcessing.empty()) {
      frame = framesForProcessing.front();
      framesForProcessing.pop();
      cout << "Processing frame size: " << frame.size() << endl;
    } else {
      return;
    }
  }

  if (frame.empty()) {
    cout << "Empty frame received for processing. Skipping." << endl;
    return;
  }

  Mat blob;
  try {
    blobFromImage(frame, blob, 1 / 255.0, Size(416, 416), Scalar(0, 0, 0), true,
                  false);
  } catch (const Exception &e) {
    cerr << "Exception caught during blobFromImage: " << e.what() << endl;
    return;
  }

  {
    lock_guard<mutex> lock(objMtx);
    net.setInput(blob);
    vector<Mat> outs;
    net.forward(outs, outNames);
    // Process detection results
    float confThreshold = 0.5;
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;
    for (size_t i = 0; i < outs.size(); i++) {
      float *data = (float *)outs[i].data;
      for (int j = 0; j < outs[i].rows; j++, data += outs[i].cols) {
        Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
        Point classIdPoint;
        double confidence;
        minMaxLoc(scores, nullptr, &confidence, nullptr, &classIdPoint);
        if (confidence > confThreshold) {
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

    vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, 0.4, indices);

    unordered_map<int, Point2f> newObjects;
    for (size_t i = 0; i < indices.size(); i++) {
      int idx = indices[i];
      Rect box = boxes[idx];
      Point2f center(box.x + box.width / 2, box.y + box.height / 2);

      int objectId = -1;
      float minDist = numeric_limits<float>::max();
      for (const auto &obj : objects) {
        float dist = norm(center - obj.second);
        if (dist < minDist) {
          minDist = dist;
          objectId = obj.first;
        }
      }

      if (minDist < 50) {
        newObjects[objectId] = center;
        objects.erase(objectId);
      } else {
        newObjects[nextObjectId++] = center;
      }

      drawPred(objectId, box.x, box.y, box.x + box.width, box.y + box.height,
               frame);
    }
    objects = newObjects;
  }

  {
    lock_guard<mutex> lock(mtx);
    framesForDisplay.push(frame.clone());
    cvProcess.notify_one();
  }

  cout << "Frame processed, current display queue size: "
       << framesForDisplay.size() << endl;
}

void processFrames(Net &net, const vector<string> &outNames) {
  unordered_map<int, Point2f> objects;
  int nextObjectId = 0;

  while (processing) {
    thread processingThreads[2]; // Increased threads back to 4
    for (int i = 0; i < 2 && processing; ++i) {
      processingThreads[i] = thread(processFrame, ref(net), ref(outNames),
                                    ref(objects), ref(nextObjectId));
    }
    for (int i = 0; i < 2; ++i) {
      if (processingThreads[i].joinable()) {
        processingThreads[i].join();
      }
    }
  }
  processing = false;
  cvProcess.notify_all();
}

void displayFrames() {
  while (processing || !framesForDisplay.empty()) {
    unique_lock<mutex> lock(mtx);
    cvProcess.wait(lock,
                   [] { return !framesForDisplay.empty() || !processing; });

    while (!framesForDisplay.empty()) {
      Mat frame = framesForDisplay.front();
      framesForDisplay.pop();
      lock.unlock();

      if (frame.empty()) {
        processing = false;
        break;
      }
      imshow("YOLO Detection", frame);
      if (waitKey(1) == 'q') {
        processing = false;
        break;
      }

      lock.lock();
    }
  }
}

int main(int argc, char **argv) {
  VideoCapture cap;
  if (argc >= 2) {
    cap.open(argv[1]);
  } else {
    cap.open(0);
  }

  if (!cap.isOpened()) {
    cout << "Cannot open video stream!" << endl;
    return -1;
  }

  Net net;
  try {
    net = readNetFromDarknet("../net/yolov3.cfg", "../net/yolov3.weights");
  } catch (const Exception &e) {
    cerr << "Exception caught during readNetFromDarknet: " << e.what() << endl;
    return -1;
  }

  net.setPreferableBackend(DNN_BACKEND_OPENCV);
  net.setPreferableTarget(DNN_TARGET_CPU);
  vector<string> outNames = net.getUnconnectedOutLayersNames();

  thread capThread(captureFrames, ref(cap));
  thread procThread(processFrames, ref(net), ref(outNames));

  displayFrames(); // Run display on main thread

  capThread.join();
  capturing = false;
  processing = false;
  cvCapture.notify_all();
  cvProcess.notify_all();
  procThread.join();

  return 0;
}
