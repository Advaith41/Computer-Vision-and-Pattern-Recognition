#pragma once
/*
* Creator - Shreejit Gajanan Deshmukh, Venkata Sai Advaith Kandiraju
* Assignment 4 - Augmented Reality Part B - Projecting objects on the image plane
* Pattern Recognition and Computer Vision
*/

#include<iostream>
#include<opencv2/opencv.hpp>
#include<fstream>
#include <opencv2/highgui/highgui_c.h>

using namespace std;
using namespace cv;

//Loads the intrinsic camera matrix and distortion coefficients found in part A
int Load_csv(cv::Mat& camera_matrix, cv::Mat& distCoeffs) {
	ifstream file("Camera_Parameters.csv");
	string line;
	vector<string> all_paras;
	vector<double> all_paras_f;
	
	//loading all parameters into a vector of strings
	while (getline(file, line)) {
		stringstream s(line);
		string para;
		while (getline(s, para, ',')) {
			all_paras.push_back(para);
		}
	}
	
	//converting all elemtents to floats
	for (int i = 0; i < all_paras.size(); i++) {

		double conv = stod(all_paras[i]);
		all_paras_f.push_back(conv);
	}
	cout << "Number of parameters " << all_paras_f.size() << endl;

	//updating the camera matrix
	camera_matrix.at<double>(0, 0) = all_paras_f[0];
	camera_matrix.at<double>(0, 1) = all_paras_f[1];
	camera_matrix.at<double>(0, 2) = all_paras_f[2];
	camera_matrix.at<double>(1, 0) = all_paras_f[3];
	camera_matrix.at<double>(1, 1) = all_paras_f[4];
	camera_matrix.at<double>(1, 2) = all_paras_f[5];
	camera_matrix.at<double>(2, 0) = all_paras_f[6];
	camera_matrix.at<double>(2, 1) = all_paras_f[7];
	camera_matrix.at<double>(2, 2) = all_paras_f[8];

	//updating the distortion matrix
	for (int i = 0; i < 5; i++) {
		distCoeffs.at<double>(i,0) = all_paras_f[9 + i];
	}
	cout << "Parameters loaded successfully" << endl;
	return 0;
}

//Function which takes in input as grayscale and changes the corners vector with the inner corners present in the image
void Find_Corners(Mat& gray, vector<Point2f>& corners) {
	Size patternsize(9, 6); //interior number of corners
	//check if checker board is present
	//try remove adap_thresh
	bool patternfound = findChessboardCorners(gray, patternsize, corners,
		CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE
		+ CALIB_CB_FAST_CHECK);
	if (patternfound) {
		cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
		//cout << "Corners found" << endl;
	}
	else {
		//cout << "Pattern not found" << endl;
	}
}