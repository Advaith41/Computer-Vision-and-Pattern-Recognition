#include <opencv2/opencv.hpp>
#include <iostream>

#include "Filter.h"

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {
    cv::VideoCapture* capdev;
    int f = 0;
    Mat output;
    int levels=10;
    int magThreshold = 10;
    int stuff = 0;
    capdev = new cv::VideoCapture(0);
    cout << "Welcome to my effects program\n This program can apply various effects to your live stream\n Press s for saving the image and q to quit\n It will revert back to original live stream upon pressing n\n Press g for Grayscale video\n Press h for an alternate Grayscale video\n Press b to Blur\n Press x to Apply sobel X\n Press y to apply Sobel Y\n Press m to apply Magnitude threshold\n Press l to apply Gradient levels\n Press c to apply Cartoon effect\n Press f to flip the stream horizontally\n Press j for special Avatar filter\n Press k for red to blue convertor\n Press u for Colormap effect\n";
    if (!capdev->isOpened()) {
        printf("Unable to open video device\n");
        return(-1);
    }


    cv::Size refS((int)capdev->get(cv::CAP_PROP_FRAME_WIDTH),
        (int)capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
    //printf("Expected size: %d %d\n", refS.width, refS.height);

    cv::namedWindow("MyVideo", 1);
    cv::Mat frame;

    for (;;) {
        *capdev >> frame;
        if (frame.empty()) {
            printf("frame is empty\n");
            break;
        }
        if (f == 0) {
            output = frame;
        }

        char key = cv::waitKey(10);
        if (key == 'q') {
            break;
        }
        
        if (key == 'g') {
            std::cout << "Converting to Grayscale\n";
            f = 3;
        }

        if (key == 'h') {
            cv::Mat output;
            std::cout << "Converting to alternate grayscale\n" << std::endl;
            mainalt(frame, output);
            f = 4;
        }
        if (key == 'b') {
            cout << "Gaussian Blur\n";
            f = 5;
        }
    
        if (key == 'f') {
            cout << "Horizontal Flip\n";
            f = 6;
        }
                
        if (key == 'x') {
            cout << "Sobel X\n";
            f = 7;
        }
        if (key == 'y') {
            cout << "Sobel Y\n";
            f = 8;
        }
        if (key == 'm') {
            cout << "Gradient Magnitude\n";
            f = 9;
        }
        if (key == 'l') {
            
            cout << "Please specify levels\n";
            cin >> levels;
            f = 10;
        }
        if (key == 'c') {
            cout << "Please specify a magnitude thrshold between 10 and 20\n";
            cin >> magThreshold;
            cout << "Please specify levels\n";
            cin >> levels;
            f = 11;
        }
        if (key == 'j') {
            cout << "Applying special effect AVATAR filter\n";
            f = 13;
        }
        if (key == 'k') {
            cout << "Applying special effect Red to Blue convertor\n";
            f = 14;
        }
        if (key == 'u') {
            cout << "Colormap effect\n";
                f = 15;
        }
        if (f == 3) {
            cv::cvtColor(frame, output, COLOR_RGB2GRAY);
        }
        if (f == 4) {
            mainalt(frame, output);
        }
        if (f == 5) {
            blur5x5(frame, output);
        }
        if (f == 6) {
            mainmirror(frame, output);
        }
        if (f == 7) {
            Mat A;
            sobelX3x3(frame, A);
            convertScaleAbs(A, output);
        }
        if (f == 8) {
            Mat B;
            sobelY3x3 (frame,B);
            convertScaleAbs(B, output);
        }
        if (f == 9) {
            Mat A;
            Mat B;
            Mat C;
            sobelX3x3(frame, A);
            sobelY3x3(frame, B);
            magnitude(A, B, C);
            convertScaleAbs(C, output);
        }
        if (f == 10) {
            
            blurQuantisize(frame, output, levels);
        }
        if (f == 11) {
            cartoon(frame, output, levels, magThreshold);
        }
        if (f == 12) {
            vermirror(frame, output);
        }
        if (f == 13) {
            avatar(frame, output);
        }
        if (f == 14) {
            colormap(frame, output);
        }
        if (f == 15) {
            cv::applyColorMap(frame, output, cv::COLORMAP_JET);
        }
        if (key == 'n') {
            f = 0;
        }
        if (key == 's') {
            cout << "Saving Image\n";
            imwrite("Captured_image.jpg", output);
        }
        imshow("MyVideo", output);

     }
    


    delete capdev;
    return(0);
}
