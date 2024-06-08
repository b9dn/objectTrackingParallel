
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <queue>
#include <thread>
#include <unordered_map>

using namespace std;
using namespace cv;
using namespace dnn;

// Global queues
queue<Mat> frameQueue;
queue<pair<Mat, vector<Rect>>> detectionQueue;
queue<pair<Mat, unordered_map<int, Point2f>>> trackingQueue;

// Synchronization primitives
mutex mtxFrame, mtxDetection, mtxTracking;
condition_variable cvFrame, cvDetection, cvTracking;
bool stopFlag = false;

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

void frameCapture(VideoCapture &cap) {
  while (!stopFlag) {
    Mat frame;
    cap >> frame;
    if (frame.empty())
      break;

    unique_lock<mutex> lock(mtxFrame);
    frameQueue.push(frame);
    cvFrame.notify_one();
  }
}

void objectDetection(Net &net, vector<string> &outNames) {
  while (!stopFlag) {
    Mat frame;
    {
      unique_lock<mutex> lock(mtxFrame);
      cvFrame.wait(lock, [] { return !frameQueue.empty() || stopFlag; });
      if (stopFlag)
        break;
      frame = frameQueue.front();
      frameQueue.pop();
    }

    Mat blob;
    blobFromImage(frame, blob, 1 / 255.0, Size(416, 416), Scalar(0, 0, 0), true,
                  false);
    net.setInput(blob);

    vector<Mat> outs;
    net.forward(outs, outNames);

    float confThreshold = 0.5, nmsThreshold = 0.4;
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;
    for (size_t i = 0; i < outs.size(); i++) {
      float *data = (float *)outs[i].data;
      for (int j = 0; j < outs[i].rows; j++, data += outs[i].cols) {
        Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
        Point classIdPoint;
        double confidence;
        minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
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

    // Apply NMS to eliminate redundant overlapping boxes with lower confidences
    vector<int> nmsIndices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, nmsIndices);
    vector<Rect> nmsBoxes;
    for (auto idx : nmsIndices) {
      nmsBoxes.push_back(boxes[idx]);
    }

    {
      lock_guard<mutex> lock(mtxDetection);
      detectionQueue.push({frame, nmsBoxes});
      cvDetection.notify_one();
    }
  }
}

void objectTracking() {
  unordered_map<int, Point2f> objects;
  int nextObjectId = 0;

  while (!stopFlag) {
    Mat frame;
    vector<Rect> boxes;

    {
      unique_lock<mutex> lock(mtxDetection);
      cvDetection.wait(lock,
                       [] { return !detectionQueue.empty() || stopFlag; });
      if (stopFlag)
        break;

      tie(frame, boxes) = detectionQueue.front();
      detectionQueue.pop();
    }

    unordered_map<int, Point2f> newObjects;
    for (const auto &box : boxes) {
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

    {
      lock_guard<mutex> lock(mtxTracking);
      trackingQueue.push({frame, objects});
      cvTracking.notify_one();
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
    cout << "Nie można otworzyć strumienia wideo!" << endl;
    return -1;
  }

  Net net = readNetFromDarknet("../net/yolov3.cfg", "../net/yolov3.weights");
  net.setPreferableBackend(DNN_BACKEND_OPENCV);
  net.setPreferableTarget(DNN_TARGET_CPU);

  vector<string> outNames = net.getUnconnectedOutLayersNames();

  thread t1(frameCapture, ref(cap));
  thread t2(objectDetection, ref(net), ref(outNames));
  thread t3(objectTracking);

  while (!stopFlag) {
    Mat frame;
    unordered_map<int, Point2f> objects;

    {
      unique_lock<mutex> lock(mtxTracking);
      cvTracking.wait(lock, [] { return !trackingQueue.empty() || stopFlag; });
      if (stopFlag)
        break;

      tie(frame, objects) = trackingQueue.front();
      trackingQueue.pop();
    }

    imshow("YOLO Detection", frame);
    if (waitKey(30) == 'q') {
      stopFlag = true;
      cvFrame.notify_all();
      cvDetection.notify_all();
      cvTracking.notify_all();
    }
  }

  t1.join();
  t2.join();
  t3.join();

  return 0;
}
