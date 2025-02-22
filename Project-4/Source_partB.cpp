
/*
* Creator - Shreejit Gajanan Deshmukh, Venkata Sai Advaith Kandiraju
* Assignment 4 - Augmented Reality Part B - Projecting objects on the image plane
* Pattern Recognition and Computer Vision
*/

#include<iostream>
#include<opencv2/opencv.hpp>
#include"Header_partB.h"

using namespace cv;
using namespace std;

int main(int argc, char* argv[]) {
	Mat camera_matrix = Mat::zeros(cv::Size(3, 3), CV_64FC1); //for storing camera intrinsic matrix
	Mat distCoeffs = Mat::zeros(5, 1, CV_64F); //for storing distortion coefficients
	vector<Point3f> points;  //this will be filled with 3D points

	VideoCapture* vidsrc;
	Mat frame;

	//function which loads our intrinsic camera matrix and distortion coefficients found in partA
	Load_csv(camera_matrix, distCoeffs);

	//Making 3D point list
	for (int i = 0; i < 6; i++) {
		for (int j = 0; j < 9; j++) {
			Point3f P(-i, j, 0);
			points.push_back(P);
		}
	}

	//capturing video via webcam
	vidsrc = new cv::VideoCapture(0);

	//camera testing
	if (!vidsrc->isOpened()) {
		printf("Unable to open video device\n");
		return(-1);
	}

	// properties of video feed
	cv::Size refS((int)vidsrc->get(cv::CAP_PROP_FRAME_WIDTH),
		(int)vidsrc->get(cv::CAP_PROP_FRAME_HEIGHT));
	printf("Expected size: %d %d\n", refS.width, refS.height);

	cv::namedWindow("Video", 1); // identifies a window

	cout << "'o' to see rotation and translation vectors live for a checker_board" << endl
		<< "'p' to project end points and 3D axis" << endl
		<< "'b' to project a 3D object" << endl;

	int flag = 0;

	for (;;) {
		//check for getting corners
		int check_2 = 0;

		Mat out_frame, rvecs, tvecs;

		char key = cv::waitKey(10);
		*vidsrc >> frame; // get a new frame from the camera, treat as a stream

		Mat frame_gray;
		cvtColor(frame, frame_gray, COLOR_BGR2GRAY);

		vector<Point2f> corners; //this will be filled by the detected corners
		
		//camera check
		if (frame.empty()) {
			printf("frame is empty\n");
			break;
		}
		Size frame_size = frame.size();

		if (key == 'd') {
			flag = 0;
		}
		if (key == 'o') {
			flag = 1;
		}
		if (key == 'p') {
			flag = 2;
		}
		if (key == 'b') {
			flag = 3;
		}
		if (key == 'q') {
			break;
		}
		if (flag == 1 or flag == 2 or flag == 3) {
			Find_Corners(frame_gray, corners);
			if (!corners.empty()) {
				//check if rvec and tvec are successfully found
				bool success = solvePnP(points, corners, camera_matrix, distCoeffs, rvecs, tvecs, false, SOLVEPNP_ITERATIVE);
				if (success) {
					check_2 = 1;
				}
			}
		}
		//default check value
		if (check_2 == 0) {
			flag = 0;			//just show the frame as it is
		}
		
		//default frame to be shown
		if (flag == 0) {
			out_frame = frame;
		}

		//task 4
		if (flag == 1 && check_2 == 1) {
			cout << "The rotation matrix is - " << rvecs << endl;
			cout << "The translation vector is - " << tvecs << endl;
			out_frame = frame;
		}

		//task 5
		if (flag == 2 && check_2 == 1) {
			//projecting corner points
			vector<Point3f> corner_pts;
			vector<Point2f> image_pts;
			corner_pts.push_back(Point3f(0, 0, 0));
			corner_pts.push_back(Point3f(0, 8, 0));
			corner_pts.push_back(Point3f(-5, 0, 0));
			corner_pts.push_back(Point3f(-5, 8, 0));

			projectPoints(corner_pts, rvecs, tvecs, camera_matrix, distCoeffs, image_pts);
			for (Point2f P : image_pts) {
				circle(frame, P, 1, Scalar(255, 0, 0));
			}
			out_frame = frame;
			//projecting co-ord frame
			bool co_ord = 1;
			if (co_ord) {
				vector<Point3f> co_ord_pts;
				vector<Point2f> image_cords;
				co_ord_pts.push_back(Point3f(0, 0, 0));
				co_ord_pts.push_back(Point3f(0, 5, 0));
				co_ord_pts.push_back(Point3f(-5, 0, 0));
				co_ord_pts.push_back(Point3f(0, 0, -5));

				projectPoints(co_ord_pts, rvecs, tvecs, camera_matrix, distCoeffs, image_cords);
				//x co-ord line
				arrowedLine(frame, image_cords[0], image_cords[1], Scalar(255, 0, 0));
				//y co-ord line
				arrowedLine(frame, image_cords[0], image_cords[2], Scalar(0, 255, 0));
				//z co-ord line
				arrowedLine(frame, image_cords[0], image_cords[3], Scalar(0, 0, 255));
				out_frame = frame;
			}
		}

		//task 6
		int counter = 0;	//for counting number of lines
		if (flag == 3 && check_2 == 1) {
			//making a star
			vector<Point3f> star_3d;
			vector<Point2f> star_2d;

			//drawing a star first
			star_3d.push_back(Point3f(0, 3, 0)); //0
			star_3d.push_back(Point3f(-1, 4, 0));
			star_3d.push_back(Point3f(-1, 5, 0));
			star_3d.push_back(Point3f(-2, 4, 0));
			star_3d.push_back(Point3f(-3, 5, 0));
			star_3d.push_back(Point3f(-2.25, 3, 0)); //5
			star_3d.push_back(Point3f(-3, 1, 0));
			star_3d.push_back(Point3f(-2, 2, 0));
			star_3d.push_back(Point3f(-1, 1, 0));
			star_3d.push_back(Point3f(-1, 2, 0));
			star_3d.push_back(Point3f(0, 3, 0)); //10

			//drawing outer star
			star_3d.push_back(Point3f(0, 3, -3)); //11
			star_3d.push_back(Point3f(-1, 4, -3));
			star_3d.push_back(Point3f(-1, 5, -3));
			star_3d.push_back(Point3f(-2, 4, -3));
			star_3d.push_back(Point3f(-3, 5, -3));
			star_3d.push_back(Point3f(-2.25, 3, -3)); //16
			star_3d.push_back(Point3f(-3, 1, -3));
			star_3d.push_back(Point3f(-2, 2, -3));
			star_3d.push_back(Point3f(-1, 1, -3));
			star_3d.push_back(Point3f(-1, 2, -3));
			star_3d.push_back(Point3f(0, 3, -3)); //21

			//drawing edges
			star_3d.push_back(Point3f(0, 3, 0)); //22
			star_3d.push_back(Point3f(0, 3, -3));
			
			star_3d.push_back(Point3f(-1, 4, 0));
			star_3d.push_back(Point3f(-1, 4, -3));
			
			star_3d.push_back(Point3f(-1, 5, 0));
			star_3d.push_back(Point3f(-1, 5, -3));
			
			star_3d.push_back(Point3f(-2, 4, 0)); //28
			star_3d.push_back(Point3f(-2, 4, -3));
			
			star_3d.push_back(Point3f(-3, 5, 0));
			star_3d.push_back(Point3f(-3, 5, -3));
			
			star_3d.push_back(Point3f(-2.25, 3, 0));
			star_3d.push_back(Point3f(-2.25, 3, -3));
			
			star_3d.push_back(Point3f(-3, 1, 0)); //34
			star_3d.push_back(Point3f(-3, 1, -3));
			
			star_3d.push_back(Point3f(-2, 2, 0));
			star_3d.push_back(Point3f(-2, 2, -3));
			
			star_3d.push_back(Point3f(-1, 1, 0));
			star_3d.push_back(Point3f(-1, 1, -3));
			
			star_3d.push_back(Point3f(-1, 2, 0)); //40
			star_3d.push_back(Point3f(-1, 2, -3));
			
			star_3d.push_back(Point3f(0, 3, 0));
			star_3d.push_back(Point3f(0, 3, -3)); //43
			//cout << star_3d.size() << endl;

			projectPoints(star_3d, rvecs, tvecs, camera_matrix, distCoeffs, star_2d);
			int n = 0;
			while (n < star_3d.size()) {
				//lower star
				if (n < 11) {
					line(frame, star_2d[n], star_2d[n + 1], Scalar(0, 0, 255));
					n += 1;
				}
				//upper star
				else if (n < 22) {
					line(frame, star_2d[n], star_2d[n + 1], Scalar(0, 0, 255));
					n += 1;
				}
				//intermediate lines
				else {
					line(frame, star_2d[n], star_2d[n + 1], Scalar(0, 0, 255));
					n += 2;
				}
			}
			out_frame = frame;
		}
		imshow("Video", out_frame);
	}
	destroyAllWindows();
	return 0;
}