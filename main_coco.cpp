#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;
using namespace dnn;

void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame) {
    // Rysowanie prostokąta wokół obiektu
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 3);

    // Ładowanie nazw klas z pliku coco.names
    static vector<string> classes;
    if (classes.empty()) {
        ifstream ifs("../net/coco.names");
        string line;
        while (getline(ifs, line)) {
            classes.push_back(line);
        }
    }

    // Dodawanie etykiety
    string label = format("%.2f", conf);
    if (!classes.empty()) {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ":" + label;
    }

    // Wyświetlanie etykiety i tła etykiety
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - round(1.5*labelSize.height)), 
              Point(left + round(1.5*labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 1);
}

int main(int argc, char** argv) {
    // Użycie: program <ścieżka do wideo> (opcjonalne)
    VideoCapture cap;
    if (argc == 2) {
        cap.open(argv[1]);
    } else {
        cap.open(0); // Domyślnie otwiera kamerę
    }

    if (!cap.isOpened()) {
        cout << "Nie można otworzyć strumienia wideo!" << endl;
        return -1;
    }

    // Wczytywanie modelu YOLO
    Net net = readNetFromDarknet("../net/yolov3.cfg", "../net/yolov3.weights");
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    // Lista warstw wyjściowych
    vector<string> outNames = net.getUnconnectedOutLayersNames();

    Mat frame, blob;
    while (cap.read(frame)) {
        blobFromImage(frame, blob, 1/255.0, Size(416, 416), Scalar(0,0,0), true, false);
        net.setInput(blob);

        // Wykonywanie detekcji
        vector<Mat> outs;
        net.forward(outs, outNames);

        // Przetwarzanie wyników detekcji
        float confThreshold = 0.5;
        vector<int> classIds;
        vector<float> confidences;
        vector<Rect> boxes;
        for (size_t i = 0; i < outs.size(); ++i) {
            float* data = (float*)outs[i].data;
            for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
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

        // NMS (Non-maximum suppression)
        vector<int> indices;
        NMSBoxes(boxes, confidences, confThreshold, 0.4, indices);
        for (size_t i = 0; i < indices.size(); ++i) {
            int idx = indices[i];
            Rect box = boxes[idx];
            drawPred(classIds[idx], confidences[idx], box.x, box.y,
                     box.x + box.width, box.y + box.height, frame);
        }

        // Wyświetlanie klatki
        imshow("YOLO Detection", frame);

        // Wyjście przy naciśnięciu 'q'
        if (waitKey(30) == 'q') break;
    }

    return 0;
}

