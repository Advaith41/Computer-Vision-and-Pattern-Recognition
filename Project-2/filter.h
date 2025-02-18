/*
Creator - Shreejit Deshmukh
Pattern Recognization and Computer Vision Spring 2023
Live Video Filtering
*/
#pragma once
#include <opencv2/opencv.hpp>
#include <cmath>

int greyscale(cv::Mat& src, cv::Mat& dst) {
	dst = cv::Mat::zeros(src.size(), CV_8UC1);
	for (int i = 0; i < src.rows; i++) {
		cv::Vec3b *rptr = src.ptr<cv::Vec3b>(i);  // rptr - src row ptr

		uchar* drptr = dst.ptr<uchar>(i); //dptr - destination row ptr
		
		for (int j = 0; j < src.cols; j++) {
			drptr[j] = (rptr[j][0] + rptr[j][1] + rptr[j][2])/3;
		}
	}
	
	return (0);
}

int blur5x5(cv::Mat& src, cv::Mat& dst) {
	cv::Mat dst_c;
	src.copyTo(dst_c);	//so that we have same 1st and last rows and columns

	//Implementing seperable gaussian filtering 1st row wise [1 2 4 2 1] then column wise [1 2 4 2 1]T
//	for (int x = 0; x < n; x++) {
		for (int i = 2; i < src.rows - 2; i++) {
			cv::Vec3b* r_sptr = src.ptr<cv::Vec3b>(i);
			cv::Vec3b* r_dptr = dst_c.ptr<cv::Vec3b>(i);

			for (int j = 2; j < src.cols - 2; j++) {
				for (int c = 0; c < 3; c++) {
					r_dptr[j][c] = (r_sptr[j - 2][c] * 1 + r_sptr[j - 1][c] * 2 + r_sptr[j][c] * 4 + r_sptr[j + 1][c] * 2 + r_sptr[j + 2][c] * 1)/10;
				}
			}
		}

		cv::Mat dst_c2;
		dst_c.copyTo(dst_c2);

		for (int i = 2; i < src.rows - 2; i++) {
			cv::Vec3b* r_sptr = dst_c.ptr<cv::Vec3b>(i);
			cv::Vec3b* r_dptr = dst_c2.ptr<cv::Vec3b>(i);
			//defining top 2 rows and bottom 2 rows for row of interest
			cv::Vec3b* r_sm2ptr = dst_c.ptr<cv::Vec3b>(i - 2);	//-2nd row
			cv::Vec3b* r_sm1ptr = dst_c.ptr<cv::Vec3b>(i - 1);	//-1st row
			cv::Vec3b* r_sp1ptr = dst_c.ptr<cv::Vec3b>(i + 1);	//+1st row
			cv::Vec3b* r_sp2ptr = dst_c.ptr<cv::Vec3b>(i + 2);		//+2nd row

			for (int j = 2; j < src.cols - 2; j++) {
				for (int c = 0; c < 3; c++) {
					r_dptr[j][c] = (r_sm2ptr[j][c] * 1 + r_sm1ptr[j][c] * 2 + r_sptr[j][c] * 4 + r_sp1ptr[j][c] * 2 + r_sp2ptr[j][c] * 1)/10; //divided by 10 as it can be the max value
				}
			}
		}
		cv::convertScaleAbs(dst_c2, dst);
;	return(0);
}

//Implementing sobel filter in x direction [-1 0 1] in row & [1 2 1]T in column
int sobelX3x3(cv::Mat& src, cv::Mat& dst) {
	cv::Mat dst_i = cv::Mat::zeros(src.size(), CV_16SC3);	//using short as values can be negative

	for (int i = 1; i < src.rows - 1; i++) {
		cv::Vec3b* r_sptr = src.ptr<cv::Vec3b>(i);
		cv::Vec3s* r_dptr = dst_i.ptr<cv::Vec3s>(i);

		for (int j = 1; j < src.cols - 1; j++) {
			for (int c = 0; c < 3; c++) {
				r_dptr[j][c] = (r_sptr[j - 1][c] * -1 + r_sptr[j + 1][c] * 1);
			}
		}
	}

	dst = cv::Mat::zeros(src.size(), CV_16SC3);	//intermediate matrix

	for (int i = 1; i < src.rows - 1; i++) {
		cv::Vec3s* r_sptr = dst_i.ptr<cv::Vec3s>(i);
		cv::Vec3s* r_dptr = dst.ptr<cv::Vec3s>(i);
		cv::Vec3s* r_sm1ptr = dst_i.ptr<cv::Vec3s>(i - 1);	//-1st row
		cv::Vec3s* r_sp1ptr = dst_i.ptr<cv::Vec3s>(i + 1);	//+1st row

		for (int j = 1; j < src.cols - 1; j++) {
			for (int c = 0; c < 3; c++) {
				r_dptr[j][c] = (r_sm1ptr[j][c] * 1 + r_sptr[j][c] * 2 + r_sp1ptr[j][c] * 1)/4; //dividing by 4 as it can be the max value
			}
		}
	}
	return (0);
}
//Sobel filter in Y direction [1 2 1] in row [1 0 -1]T in column
int sobelY3x3(cv::Mat& src, cv::Mat& dst) {
	cv::Mat dst_i = cv::Mat::zeros(src.size(), CV_16SC3);	//short is used as values can be negative
	

	for (int i = 1; i < src.rows - 1; i++) {
		cv::Vec3b* r_sptr = src.ptr<cv::Vec3b>(i);
		cv::Vec3s* r_dptr = dst_i.ptr<cv::Vec3s>(i);

		for (int j = 1; j < src.cols - 1; j++) {
			for (int c = 0; c < 3; c++) {
				r_dptr[j][c] = (r_sptr[j - 1][c] * 1 + r_sptr[j][c] * 2 + r_sptr[j + 1][c] * 1)/4;
			}
		}
	}

	dst = cv::Mat::zeros(src.size(), CV_16SC3);	//intermediate matrix

	for (int i = 1; i < src.rows - 1; i++) {
		cv::Vec3s* r_sptr = dst_i.ptr<cv::Vec3s>(i);
		cv::Vec3s* r_dptr = dst.ptr<cv::Vec3s>(i);
		cv::Vec3s* r_sm1ptr = dst_i.ptr<cv::Vec3s>(i - 1);	//-1st row
		cv::Vec3s* r_sp1ptr = dst_i.ptr<cv::Vec3s>(i + 1);	//+1st row

		for (int j = 1; j < src.cols - 1; j++) {
			for (int c = 0; c < 3; c++) {
				r_dptr[j][c] = (r_sm1ptr[j][c] * -1 + r_sp1ptr[j][c] * 1);
			}
		}
	}
	return (0);
}

int magnitude(cv::Mat& sx, cv::Mat& sy, cv::Mat& dst) {
	dst = cv::Mat::zeros(sx.size(), CV_16SC3);
	for (int i = 0; i < sx.rows; i++) {
		cv::Vec3s* rsxptr = sx.ptr<cv::Vec3s>(i);
		cv::Vec3s* rsyptr = sy.ptr<cv::Vec3s>(i);

		cv::Vec3s* dptr = dst.ptr<cv::Vec3s>(i);
		for (int j = 0; j < sx.cols; j++) {
			for (int c = 0; c < 3; c++) {
				dptr[j][c] = std::sqrt(rsxptr[j][c]*rsyptr[j][c] + rsyptr[j][c]*rsyptr[j][c]);
			}
		}
	}
	return (0);
}

int blurQuantisize(cv::Mat & src, cv::Mat & dst, int level) {
	src.copyTo(dst);
	for (int i = 0; i < src.rows; i++) {
		cv::Vec3b* rptr = src.ptr<cv::Vec3b>(i);

		cv::Vec3b* dptr = dst.ptr<cv::Vec3b>(i);
		int divider = 255 / level;
		for (int j = 0; j < src.cols; j++) {
			for (int c = 0; c < 3; c++) {

				dptr[j][c] = (int)((int)(rptr[j][c] / divider) * divider);
			}
		}
	}
	return(0);
}

int cartoon(cv::Mat& src, cv::Mat& dst, int levels, int magThreshold) {
	cv::Mat s1, s2, mag;
	sobelX3x3(src, s1);
	sobelY3x3(src, s2);
	magnitude(s1, s2, mag);
	blurQuantisize(src, dst, levels);

	for (int i = 0; i < dst.rows; i++) {
		cv::Vec3s* mptr = mag.ptr<cv::Vec3s>(i); //pointer pointing to Sobel X-Y magnitude filtered frame
		
		cv::Vec3s* dptr = dst.ptr<cv::Vec3s>(i); //pointer pointing to the blurred input image

		for (int j = 0; j < src.cols; j++) {
			for (int c = 0; c < 3; c++) {
				if (mptr[j][c] > 255 - magThreshold) {
					dptr[j][0] = 0;					//setting pixel to 0 value whereever it is less than threshold
					dptr[j][1] = 0;
					dptr[j][2] = 0;
				}
			}
		}
	}
	return (0);
}

int negative(cv::Mat& src, cv::Mat& dst) {
	src.copyTo(dst);
	for (int i = 0; i < src.rows; i++) {
		cv::Vec3b* rptr = src.ptr<cv::Vec3b>(i);

		cv::Vec3b* dsptr = dst.ptr<cv::Vec3b>(i);
		for (int j = 0; j < src.cols; j++) {
			for (int c = 0; c < 3; c++) {
				dsptr[j][c] = 255 - rptr[j][c];
			}
		}
	}
	return(0);
}

int whiteboi(cv::Mat& src, cv::Mat& dst) {
	cv::Mat src_hsv;
	cv::cvtColor(src, src_hsv, cv::COLOR_BGR2HSV);
	src.copyTo(dst);
	for (int i = 0; i < src.rows; i++) {
		cv::Vec3b* rptr = src_hsv.ptr<cv::Vec3b>(i);
		cv::Vec3b* dptr = dst.ptr<cv::Vec3b>(i);

		for (int j = 0; j < src.cols; j++) {
			if (rptr[j][0] > 0 and rptr[j][0] < 20 and rptr[j][1] > 10 and rptr[j][1] < 150 and rptr[j][2] > 60 and rptr[j][2] < 255) {
				for (int c = 0; c < 3; c++) {
					dptr[j][c] = 255;
				}
			}
			else {
				for (int c = 0; c < 3; c++) {
					dptr[j][c] = 0;
				}
			}
		}
	}
	return (0);
}
