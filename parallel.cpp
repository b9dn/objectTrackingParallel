#include <condition_variable>
#include <iostream>
#include <mutex>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <queue>
#include <thread>
#include <utility>

using namespace cv;
using namespace dnn;
using namespace std;

// Global queues and synchronization variables
queue<pair<Mat, int>> frameQueue;
queue<pair<pair<Mat, int>, vector<Mat>>> outputQueue;
mutex mtx_frameQueue;
mutex mtx_outputQueue;
condition_variable cv_frameQueue;
condition_variable cv_outputQueue;
bool finished = false;
const size_t MAX_QUEUE_SIZE = 30;

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

void processFrame(const String &modelConfiguration, const String &modelWeights,
                  vector<String> outputLayers) {
  Net net = readNetFromDarknet(modelConfiguration, modelWeights);
  net.setPreferableBackend(DNN_BACKEND_OPENCV);
  net.setPreferableTarget(DNN_TARGET_CPU);

  while (true) {
    pair<Mat, int> cap;
    Mat frame;
    int frameNum;
    {
      unique_lock<mutex> lock(mtx_frameQueue);
      cv_frameQueue.wait(lock, [] { return !frameQueue.empty() || finished; });

      if (finished && frameQueue.empty())
        break;

      cap = frameQueue.front();
      frame = cap.first;
      frameNum = cap.second;
      frameQueue.pop();
    }

    Mat blob = blobFromImage(frame, 0.00392, Size(416, 416), Scalar(0, 0, 0),
                             true, false);
    net.setInput(blob);
    vector<Mat> outs;
    net.forward(outs, outputLayers);

    {
      unique_lock<mutex> lock(mtx_outputQueue);
      outputQueue.push(make_pair(make_pair(frame, frameNum), outs));
    }
    cv_outputQueue.notify_one();
  }
}

int main(int argc, char **argv) {
  // Load YOLO
  String modelConfiguration = "../net/yolov3.cfg";
  String modelWeights = "../net/yolov3.weights";

  Net net = readNetFromDarknet(modelConfiguration, modelWeights);
  vector<String> layerNames = net.getLayerNames();
  vector<String> outputLayers;
  for (auto &i : net.getUnconnectedOutLayers())
    outputLayers.push_back(layerNames[i - 1]);

  // Open the camera or video file
  VideoCapture cap;
  string source =
      argc > 1 ? argv[1] : "0"; // Default to "0" if no argument is given
  if (source == "0") {
    cap.open(0); // Open default camera
  } else {
    cap.open(source); // Open video file or camera by identifier
  }

  if (!cap.isOpened()) {
    cerr << "Error: Unable to open the video source: " << source << endl;
    return -1;
  }

  double fps = cap.get(CAP_PROP_FPS);
  int frameWidth = cap.get(CAP_PROP_FRAME_WIDTH);
  int frameHeight = cap.get(CAP_PROP_FRAME_HEIGHT);

  cout << "Source: " << source << "\n";
  cout << "FPS: " << fps << "\n";
  cout << "Frame Width: " << frameWidth << "\n";
  cout << "Frame Height: " << frameHeight << "\n";

  // Create threads
  vector<thread> workers;
  int numThreads = 4;
  for (int i = 0; i < numThreads; ++i)
    workers.push_back(
        thread(processFrame, modelConfiguration, modelWeights, outputLayers));

  int frameCount = 0;
  int nextObjectId = 0;
  unordered_map<int, Point2f> objects;
  int actualDisplayNum = 0;

  while (true) {
    Mat frame;
    cap >> frame;

    if (frame.empty()) {
      cerr << "Error: Frame capture failed or end of stream. Source might "
              "be "
              "an unsupported video format or the file might be corrupted."
           << endl;
      break;
    }

    frameCount++;
    cout << "Captured frame size: " << frame.cols << "x" << frame.rows
         << " | Frame Count: " << frameCount << endl;

    if (frameCount % 10 == 0) {
      {
        unique_lock<mutex> lock(mtx_frameQueue);
        if (frameQueue.size() >= MAX_QUEUE_SIZE) {
          cout << "Frame queue is full, waiting..." << endl;
          cv_frameQueue.wait(lock,
                             [] { return frameQueue.size() < MAX_QUEUE_SIZE; });
        }
        frameQueue.push(make_pair(frame, frameCount));
      }
      cv_frameQueue.notify_one();
    }

    if (outputQueue.empty())
      continue;

    pair<pair<Mat, int>, vector<Mat>> result;
    {
      unique_lock<mutex> lock(mtx_outputQueue);
      cv_outputQueue.wait(lock,
                          [] { return !outputQueue.empty() || finished; });

      if (finished && outputQueue.empty())
        break;

      result = outputQueue.front();
      outputQueue.pop();
    }

    Mat displayFrame = result.first.first;
    int displayFrameNum = result.first.second;
    vector<Mat> outs = result.second;

    if (displayFrameNum < actualDisplayNum)
      continue;

    actualDisplayNum = displayFrameNum;
    float confThreshold = 0.5;
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;
    for (size_t i = 0; i < outs.size(); ++i) {
      float *data = (float *)outs[i].data;
      for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
        Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
        Point classIdPoint;
        double confidence;
        minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
        if (confidence > 0.5) {
          int centerX = (int)(data[0] * displayFrame.cols);
          int centerY = (int)(data[1] * displayFrame.rows);
          int width = (int)(data[2] * displayFrame.cols);
          int height = (int)(data[3] * displayFrame.rows);
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

      // Znajdź najbliższy istniejący obiekt
      int objectId = -1;
      float minDist = numeric_limits<float>::max();
      for (const auto &obj : objects) {
        float dist = norm(center - obj.second);
        if (dist < minDist) {
          minDist = dist;
          objectId = obj.first;
        }
      }

      // Jeśli odległość jest mniejsza niż pewien próg, przypisz ID obiektu
      if (minDist < 50) {
        newObjects[objectId] = center;
        objects.erase(objectId);
      } else {
        // Przydziel nowy ID
        newObjects[nextObjectId++] = center;
      }

      drawPred(objectId, box.x, box.y, box.x + box.width, box.y + box.height,
               displayFrame);
    }

    objects = newObjects;
    imshow("Frame", displayFrame);
    if (waitKey(1) == 'q')
      break;
  }

  cout << "Total frames captured: " << frameCount << endl;

  // Set finish flag and notify threads
  {
    unique_lock<mutex> lock_frame(mtx_frameQueue);
    unique_lock<mutex> lock_output(mtx_outputQueue);
    finished = true;
  }
  cv_frameQueue.notify_all();
  cv_outputQueue.notify_all();

  // Join threads
  for (auto &t : workers)
    t.join();

  cap.release();
  destroyAllWindows();

  return 0;
}
