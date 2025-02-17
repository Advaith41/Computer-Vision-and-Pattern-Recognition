#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;

int mainalt(cv::Mat& src, cv::Mat& dst) {
    if (src.empty()) {
        return -1;
    }
    dst = cv::Mat(src.rows, src.cols, CV_8UC1);
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            cv::Vec3b pixel = src.at<cv::Vec3b>(i, j);
            int average = (pixel[0] + pixel[1] + pixel[2]) / 3;
            dst.at<uchar>(i, j) = average;
        }
    }
    return 0;
}


int blur5x5(cv::Mat& src, cv::Mat& dst) {
    Mat dst_c;
    dst_c = src.clone(); // Create a copy of the source image
    int rows = src.rows;
    int cols = src.cols;
    for (int i = 2; i < rows - 2; i++) {
        Vec3b* src_row_ptr = src.ptr<Vec3b>(i);
        Vec3b* dst_row_ptr = dst_c.ptr<Vec3b>(i);

        for (int j = 2; j < cols - 2; j++) {
            for (int c = 0; c < 3; c++) {
                dst_row_ptr[j][c] = (src_row_ptr[j - 2][c] * 1 + src_row_ptr[j - 1][c] * 2 + src_row_ptr[j][c] * 4 + src_row_ptr[j + 1][c] * 2 + src_row_ptr[j + 2][c] * 1) / 10;
            }
        }
    }
    src.copyTo(dst);
    
    for (int i = 2; i < rows - 2; i++) {
        Vec3b* src_row_ptr = dst_c.ptr<Vec3b>(i);
        Vec3b* dst_row_ptr = dst.ptr<Vec3b>(i);

        cv::Vec3b* src_row_min2 = dst_c.ptr<cv::Vec3b>(i - 2);
        cv::Vec3b* src_row_min1 = dst_c.ptr<cv::Vec3b>(i - 1);
        cv::Vec3b* src_row_plus1 = dst_c.ptr<cv::Vec3b>(i + 1);
        cv::Vec3b* src_row_plus2 = dst_c.ptr<cv::Vec3b>(i + 2);

        for (int j = 2; j < cols - 2; j++) {
            for (int c = 0; c < 3; c++) {
                dst_row_ptr[j][c] = (src_row_min2[j][c] * 1 + src_row_min1[j][c] * 2 + src_row_ptr[j][c] * 4 + src_row_plus1[j][c] * 2 + src_row_plus2[j][c] * 1) / 10;
            }
        }
    }
    //cv::convertScaleAbs(dst_c, dst);
    return 0;
}

int sobelX3x3(cv::Mat& src, cv::Mat& dst) {
    int rows = src.rows;
    int cols = src.cols;
    cv::Mat dst_c = cv::Mat::zeros(src.size(), CV_16SC3);   //using short as values can be negative    
    for (int i = 1; i < rows - 1; i++) {
        cv::Vec3b* src_row_ptr = src.ptr<cv::Vec3b>(i);
        cv::Vec3s* dst_row_ptr = dst_c.ptr<cv::Vec3s>(i);
            for (int j = 1; j < cols - 1; j++) {
                for (int c = 0; c < 3; c++) {
                dst_row_ptr[j][c] = (src_row_ptr[j - 1][c] * -1 + src_row_ptr[j + 1][c] * 1);
                }
            }
    }
dst_c.copyTo(dst);   
    for (int i = 1; i < rows - 1; i++) {
        cv::Vec3s* src_row_ptr = dst_c.ptr<cv::Vec3s>(i);
        cv::Vec3s* dst_row_ptr = dst.ptr<cv::Vec3s>(i);
        cv::Vec3s* src_row_min1 = dst_c.ptr<cv::Vec3s>(i - 1);         
        cv::Vec3s* src_row_plus1 = dst_c.ptr<cv::Vec3s>(i + 1);         
            for (int j = 1; j < cols - 1; j++) {
                for (int c = 0; c < 3; c++) {
                dst_row_ptr[j][c] = (src_row_min1[j][c] * 1 + src_row_ptr[j][c] * 2 + src_row_plus1[j][c] * 1) / 4;             
                }
            }
    }
    return (0);
}

int sobelY3x3(cv::Mat& src, cv::Mat& dst) {
    int rows = src.rows;
    int cols = src.cols;
    int value=0;
    cv::Mat dst_c = cv::Mat::zeros(src.size(), CV_16SC3);   //short is used as values can be negative    
    for (int i = 1; i < rows - 1; i++) {
    cv::Vec3b* src_row_ptr = src.ptr<cv::Vec3b>(i);
    cv::Vec3b* dst_row_ptr = dst_c.ptr<cv::Vec3b>(i);
    for (int j = 1; j < cols - 1; j++) {
        for (int c = 0; c < 3; c++) {
            dst_row_ptr[j][c] = (src_row_ptr[j - 1][c] * 1 + src_row_ptr[j][c] * 2 + src_row_ptr[j + 1][c] * 1) / 4;
            value = dst_row_ptr[j][c];
        }
    }
}
dst_c.copyTo(dst);     
for (int i = 1; i < rows - 1; i++) {
    cv::Vec3b* src_row_ptr = dst_c.ptr<cv::Vec3b>(i);
    cv::Vec3s* dst_row_ptr = dst.ptr<cv::Vec3s>(i);
    cv::Vec3b* src_row_min1 = dst_c.ptr<cv::Vec3b>(i - 1);         
    cv::Vec3b* src_row_plus1 = dst_c.ptr<cv::Vec3b>(i + 1);          
        for (int j = 1; j < cols - 1; j++) {
            for (int c = 0; c < 3; c++) {
            dst_row_ptr[j][c] = (src_row_min1[j][c] * -1 + src_row_plus1[j][c] * 1);
            }
        }
}
    return (0);
}
int magnitude(cv::Mat& sx, cv::Mat& sy, cv::Mat& dst) {
    int rows = sx.rows;
    int cols = sx.cols;
    dst = cv::Mat::zeros(sx.size(), CV_16SC3);
    for (int i = 0; i < rows; i++) {
        cv::Vec3s* sx_row = sx.ptr<cv::Vec3s>(i);
        cv::Vec3s* sy_row = sy.ptr<cv::Vec3s>(i);
        cv::Vec3s* dst_row_ptr = dst.ptr<cv::Vec3s>(i);
        for (int j = 0; j < cols; j++) {
            for (int c = 0; c < 3; c++) {
                dst_row_ptr[j][c] = std::sqrt(sx_row[j][c] * sy_row[j][c] + sy_row[j][c] * sy_row[j][c]);
            }
        }
    }
    return (0);
}

int blurQuantisize(cv::Mat& src, cv::Mat& dst, int levels) {
int rows = src.rows;
int cols = src.cols;
src.copyTo(dst);
    for (int i = 0; i < rows; i++) {
        cv::Vec3b* src_row = src.ptr<cv::Vec3b>(i);
        cv::Vec3b* dst_row = dst.ptr<cv::Vec3b>(i);
        int divider = 255 / levels;
        for (int j = 0; j < cols; j++) {
            for (int c = 0; c < 3; c++) {
                dst_row[j][c] = (int)((int)(src_row[j][c] / divider) * divider);
            }
        }
    }
    return(0);
}

int cartoon(cv::Mat& src, cv::Mat& dst, int levels, int magThreshold) {
int rows = dst.rows;
int cols = dst.cols;
cv::Mat p, q, mag;
sobelX3x3(src, p);
sobelY3x3(src, q);
magnitude(p, q, mag);
blurQuantisize(src, dst, levels);
    for (int i = 0; i < rows; i++) {
        cv::Vec3s* mag_ptr = mag.ptr<cv::Vec3s>(i); //Sobel X-Y magnitude filtered frame        
        cv::Vec3s* dst_ptr = dst.ptr<cv::Vec3s>(i); //Blurred image        
        for (int j = 0; j < cols; j++) {
            for (int c = 0; c < 3; c++) {
                if (mag_ptr[j][c] > 255 - magThreshold) {
                    dst_ptr[j][0] = 0;                 //Pixel = 0;                  
                    dst_ptr[j][1] = 0;
                    dst_ptr[j][2] = 0;
                }
            }
        }
    }
return (0);
}

void mainmirror(Mat& frame, Mat& output) {
    // Flip image horizontally
    cv::flip(frame, output, 1);
}

void vermirror(Mat& frame, Mat& output) {
    //Vertically flip
    cv::flip(frame, output, 0);
}

int colormap(cv::Mat& src, cv::Mat& dst) {
    dst = src.clone();
    int rows = dst.rows;
    int cols = dst.cols;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            cv::Vec3b& pixel = dst.at<cv::Vec3b>(i, j);
            int blue = pixel[0];
            int green = pixel[1];
            int red = pixel[2];
            // apply your color map effect here
            // for example: a basic grayscale color map
            pixel[0] = red;
            /*pixel[1] = intensity;*/
            pixel[2] = blue;
        }
    }
    return 0;
}

int avatar(cv::Mat& src, cv::Mat& dst) {
    int rows = src.rows;
    int cols = src.cols;
    Mat hsv;
    cvtColor(src, hsv, cv::COLOR_BGR2HSV);

    Scalar lower = cv::Scalar(0, 20, 70);
    Scalar upper = cv::Scalar(20, 255, 255);

    Mat mask;
    inRange(hsv, lower, upper, mask);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (mask.at<uchar>(i, j) == 255) {
                src.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 0, 0);
            }
        }
    }
    return 0;

}
