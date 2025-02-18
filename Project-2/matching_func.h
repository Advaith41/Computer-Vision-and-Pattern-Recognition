#pragma once
/*
  Creators - Shreejit Deshmukh, Venkata Sai Advaith Kandiraju
  Pattern recognition and Computer vision Sping 23
  Content based image ranking
*/
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <filter.h>


/************************************************************************************************************
	This function creates histogram of any given matrix using R and G chromaticity values of each RGB pixel 
	from the image. The input can be a matrix of any size while the output is a 18x18 chromaticity histogram
	of the image
 ************************************************************************************************************/

cv::Mat hist(cv::Mat mat) {
	float sum = 0;
	cv::Mat hist_mat = cv::Mat::zeros(18, 18, CV_32S);
	float norm_elem = 0.0;													//floating point for normalization
	int elem = 0;															//for counting number of elements
	for (int i = 0; i < mat.rows; i++) {
		cv::Vec3b* rptr_tar = mat.ptr<cv::Vec3b>(i);
		
		for (int j = 0; j < mat.cols; j++) {
			sum = rptr_tar[j][0] + rptr_tar[j][1] + rptr_tar[j][2];			//sum of all RGB values for each pixel
			if (sum == 0) {
				sum = 0.00001;												//to avoid division by zero error
			}
			int r = (rptr_tar[j][2] * 18) / sum;
			int g = (rptr_tar[j][0] * 18) / sum;
			hist_mat.at<int>(g, r) += 1;									//incrementing value at this row, column whenever the bin finds a value
			elem++;															//to count number of elements
		}
	}

	//normalizing histogram
	cv::Mat norm_hist = cv::Mat::zeros(hist_mat.size(), CV_32F);
	for (int i = 0; i < hist_mat.rows; i++) {
		for (int j = 0; j < hist_mat.cols; j++) {
			norm_elem = hist_mat.at<int>(i, j) / float(elem);
			norm_hist.at<float>(i, j) = norm_elem;
		}
	}
	return(norm_hist);
}


/************************************************************************************************************
	This function calculates L2 norm distances for any matrix of any size in 3 channels i.e the squared root 
	difference of each pixel value (RGB) and returns the total distance of the images. Lesser the distance 
	more similar the images. 
 ************************************************************************************************************/

float L2_norm(cv::Mat target, cv::Mat current) {
	float Squared_sum = 0;
	//std::cout << target.size;
	for (int i = 0; i < target.rows; i++) {
		cv::Vec3b* rptr_tar = target.ptr<cv::Vec3b>(i);
		cv::Vec3b* rptr_cur = current.ptr<cv::Vec3b>(i);
		for (int j = 0; j < target.cols; j++) {
			for (int c = 0; c < 3; c++) {
				Squared_sum += (rptr_cur[j][c] - rptr_tar[j][c]) * (rptr_cur[j][c] - rptr_tar[j][c]);		//squaring the differences for each value in the matrix
			}
		}
	}
	return(std::sqrt(Squared_sum));										//returning squared root of the sum
}


/************************************************************************************************************
	This function returns the intersection values between two matrices (returns the smaller value between the
	two), more the value more is the corelation with the target image
 ************************************************************************************************************/

float hist_intersection(cv::Mat target, cv::Mat current) {
	float inter_sum = 0;										//stores the sum of intersection
	float small = 0;												//small value for each bin
	//std::cout << target.size;
	for (int i = 0; i < target.rows; i++) {
		float* rptr_tar = target.ptr<float>(i);
		float* rptr_cur = current.ptr<float>(i);
		for (int j = 0; j < target.cols; j++) {
				small = rptr_cur[j] <= rptr_tar[j] ? rptr_cur[j] : rptr_tar[j];
				inter_sum += small;
		}
	}
	return(inter_sum);
}

/************************************************************************************************************
	This function is for base matching of the image. It sends a two matrices for direct pixel comparison to L2
	norm function
 ************************************************************************************************************/

float base_matching(cv::Mat target, cv::Mat current) {
	int tar_t_r, tar_b_r, tar_l_c, tar_r_c, cur_t_r, cur_b_r, cur_l_c, cur_r_c;
	//loading the center of our target and current 9X9 matrix
	tar_t_r = target.rows / 2 - 4;
	tar_b_r = target.rows / 2 + 5;
	cur_t_r = current.rows / 2 - 4;
	cur_b_r = current.rows / 2 + 5;
	tar_l_c = target.cols / 2 - 4;
	tar_r_c = target.cols / 2 + 5;
	cur_l_c = current.cols / 2 - 4;
	cur_r_c = current.cols / 2 + 5;
	float metric = L2_norm(target(cv::Range(tar_t_r,tar_b_r),cv::Range(tar_l_c,tar_r_c)),current(cv::Range(cur_t_r,cur_b_r),cv::Range(cur_l_c,cur_r_c)));
	return (metric);
}

/************************************************************************************************************
	This function is for matching rg chromaticity histograms of two images
 ************************************************************************************************************/

float histogram_matching(cv::Mat target, cv::Mat current) {
	float dist = 0;
	cv::Mat targ_hist = hist(target); cv::Mat cur_hist = hist(current);
	dist = hist_intersection(targ_hist, cur_hist);
	return (1 - dist);																//For max corelation d should be closer to 1
}																					//But as we are using shorter distance > co-relation we use 1 - d as our metric

/************************************************************************************************************
	This function is for histogram matching except the image is divided into two sections top half and bottom
	half
 ************************************************************************************************************/

float multi_hist(cv::Mat target, cv::Mat current) {
	//making sections for comparing - 4 quarters
	float dist_left_top, dist_right_top, dist_left_bot, dist_right_bot;
	dist_left_top = histogram_matching(target(cv::Range(0, target.rows / 2), cv::Range(0, target.cols / 2)), current(cv::Range(0, current.rows / 2), cv::Range(0, current.cols / 2)));
	dist_right_top = histogram_matching(target(cv::Range(0, target.rows / 2), cv::Range(target.cols / 2, target.cols)), current(cv::Range(0, current.rows / 2), cv::Range(current.cols / 2, current.cols)));
	dist_left_bot = histogram_matching(target(cv::Range(target.rows / 2, target.rows), cv::Range(0, target.cols / 2)), current(cv::Range(current.rows / 2, current.rows), cv::Range(0, current.cols / 2)));
	dist_right_bot = histogram_matching(target(cv::Range(target.rows / 2, target.rows), cv::Range(target.cols / 2, target.cols)), current(cv::Range(current.rows / 2, current.rows), cv::Range(current.cols / 2, current.cols)));
	return (0.25 * dist_left_top + 0.25 * dist_right_top + 0.25 * dist_left_bot + 0.25 * dist_right_bot);
	
}

/************************************************************************************************************
	This function is for comparing two features of query and target images - one is the rg chromaticity color 
	histogram and the other is sobel magnitude.
 ************************************************************************************************************/

float text_color(cv::Mat target, cv::Mat current) {
	float dist_text, dist_col, dist_res;
	float col_w = 0.5; int text_w = 0.5;
	cv::Mat tar_grad_image, cur_grad_image, tar_Sx, tar_Sy, cur_Sx, cur_Sy;
	cv::Mat target_hist, current_hist, tar_grad_image_hist, cur_grad_image_hist;

	//converting to sobel images
	sobelX3x3(target,tar_Sx);
	sobelX3x3(current,cur_Sx);
	sobelY3x3(target, tar_Sy);
	sobelY3x3(current,cur_Sy);
	magnitude(tar_Sx, tar_Sy, tar_grad_image);
	magnitude(cur_Sx, cur_Sy, cur_grad_image);

	//Conversion to histogram before passing in the function
	target_hist = hist(target); current_hist = hist(current); tar_grad_image_hist = hist(tar_grad_image); cur_grad_image_hist = hist(cur_grad_image);

	dist_col = histogram_matching(target_hist, current_hist);
	dist_text = histogram_matching(tar_grad_image_hist, cur_grad_image_hist);
	dist_res = dist_col * col_w + dist_text * text_w;
	return dist_text;
}

/************************************************************************************************************
	This is a custom function for comparison between the target and query image using a custom - Reward Punish
	metric. The intension of this function is to find blue sky at top of images and then rank them according
	to the percent sky present in the image.
	Following steps are happening -
	1. Using top half only for detection
	2. HSV conversion and comparing with sky blue color with reward/punish metric
 ************************************************************************************************************/

float custom_func(cv::Mat current) {
	float dist = 0;
	int elem = 0;								//total number of elements for normalization
	cv::Mat cur, cur_hsv;

	//Extracting top half -> aim is to find if blue sky is at top
	cur = current(cv::Range(0, current.rows / 2), cv::Range(0, current.cols));

	//converting to hsv as better color space to work in
	cv::cvtColor(cur, cur_hsv, cv::COLOR_BGR2HSV);
	
	for (int i = 0; i < cur_hsv.rows; i++) {
		cv::Vec3b* rptr = cur_hsv.ptr<cv::Vec3b>(i);
		for (int j = 0; j < cur_hsv.cols; j++) {
			//will compare hue value matches to blue color if yes then we increase our metric
			if (rptr[j][0] > 95 && rptr[j][0] < 115 && rptr[j][1] > 70 && rptr[j][1] < 120 && rptr[j][2] > 170) {
				dist+= 1;
			}
			//if not we will punish our metric highly
			else {
				dist -= 2;
			}
			elem++;
		}
	}
	//Normalization step1 - final dist value between (-1,1)
	if (dist >= 0) {
		dist = dist / elem;
	}
	else {
		dist = dist / (2 * elem);
	}
	dist = (dist / 2) + 0.5;			//normalization step2 - to convert it to (0,1)
	return (dist);					
}

/************************************************************************************************************
	This is extension for finding bananas (in center) in the images, It performs the following steps for calculating final
	distance metric -
	1. Extract the center of the image
	2. 1st pass of gaussian for texture feature identification
	3. 2nd pass of sobel magnitude texture feature identification
	4. Color extraction for banana
	5. Reward/punish metric for finding the banana!!!!
 ************************************************************************************************************/

float banana_finder(cv::Mat current) {
	cv::Mat blur_C, Sx_C, Sy_C, Fin_T_M, Fin_C, Fin_T_G, cur, cur_hsv, blur_T, target, Sx_T, Sy_T;
	float dist_mag = 0; float dist_color = 0; float dist_gaus = 0; float dist = 0; int elem = 0;

	//Weights adjusted so that the scales are normalized
	float w_m = 0.4; 
	float w_c = 0.2;
	float w_g = 1 - w_m - w_c;
	
	//One time operation to save reference banana modified matrices and then use it later for corelation
	/*target = cv::imread("targ.jpg");
	target = target(cv::Range(target.rows / 2 - 150, target.rows / 2 + 150), cv::Range(target.cols / 2 - 150, target.cols / 2 + 150));
	
	blur5x5(target, blur_T);

	sobelX3x3(target, Sx_T);
	sobelY3x3(target, Sy_T);
	magnitude(Sx_T, Sy_T, Fin_T_M);

	cv::imwrite("reference_gaus.jpg", blur_T);
	cv::imwrite("reference_mag.jpg", Fin_T_M);*/

	Fin_T_M = cv::imread("reference_mag.jpg", 1);
	Fin_T_G = cv::imread("reference_gaus.jpg", 1);

	if (Fin_T_G.empty() or Fin_T_G.empty()) {
		std::cout << "Reference files are not present in the .exe folder";
		return -1;
	}
	
	//Extracting center portion where banana is in the middle
	cur = current(cv::Range(current.rows / 2 - 150, current.rows / 2 + 150), cv::Range(current.cols / 2 - 150, current.cols / 2 + 150));

	//For current images
	blur5x5(cur, blur_C);

	sobelX3x3(cur, Sx_C);
	sobelY3x3(cur, Sy_C);
	magnitude(Sx_C, Sy_C, Fin_C);
	
	dist_mag = 1 - histogram_matching(Fin_C, Fin_T_M);									//we are using the reverse logic for distance metric
	dist_gaus = 1 - histogram_matching(blur_C, Fin_T_G);

	//converting to hsv as better color space to work in
	cv::cvtColor(cur, cur_hsv, cv::COLOR_BGR2HSV);

	for (int i = 0; i < cur_hsv.rows; i++) {
		cv::Vec3b* rptr = cur_hsv.ptr<cv::Vec3b>(i);
		for (int j = 0; j < cur_hsv.cols; j++) {
			//will compare hue value matches to banana yellow color if yes then we increase our metric
			if (rptr[j][0] > 50 && rptr[j][0] < 70 && rptr[j][1] > 170 && rptr[j][2] > 190) {
				dist_color += 1;
			}
			//if not we will punish our metric
			else {
				dist_color -= 2;
			}
			elem++;
		}
	}

	//normalization step1 - final dist value between (-1,1)
	if (dist_color >= 0) {
		dist_color = dist_color / elem;
	}
	else {
		dist_color = dist_color / (2 * elem);
	}
	dist_color = (dist_color / 2) + 0.5;			//normalization step2 - to convert it to (0,1)
	
	//merging the results of the two operations
	dist = dist_mag * w_m + dist_color * w_c + dist_gaus * w_g;
	return dist;
}
