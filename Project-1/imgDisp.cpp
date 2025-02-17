#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

int main(int argc, char* argv[]) {

    Mat img = imread("1.jpg");

    namedWindow("Image", 1);

    imshow("Image", img);

    while (1) {
        char key = waitKey(0);
        if (key == 'q') {
            break;
        }
    }
    return (0);
}
