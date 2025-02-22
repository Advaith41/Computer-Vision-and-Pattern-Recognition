#pragma once
/*
* Creator - Shreejit Gajanan Deshmukh, Venkata Sai Advaith Kandiraju
* Assignment 4 - Augmented Reality Part A - Calibrating camera
* Pattern Recognition and Computer Vision
*/
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui_c.h>

using namespace std;
using namespace cv;

//Function which takes in input as grayscale and updates the empty corners vector with the inner corners 
//of the checkerboard present in the image
//task1
void Find_Corners(Mat &gray, vector<Point2f> &corners) {
    Size patternsize(9, 6); //interior number of corners
    //check if checker board is present
    //try remove adap_thresh
    bool patternfound = findChessboardCorners(gray, patternsize, corners,
        CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE
        + CALIB_CB_FAST_CHECK);
    if (patternfound) {
        cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
        cout << "Corners found" << endl;
    }
    else {
        cout << "Pattern not found" << endl;
    }
}
