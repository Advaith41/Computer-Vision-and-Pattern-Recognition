/*
Creator - Shreejit Gajanan Deshmukh, Venkata Sai Advaith Kandiraju
PRCV spring 23' Assignment 4 - extension
*/

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

/*
This program finds the 'X' cross sign on a piece on paper and then warps
a pikachu (Pokemon cartoon) character onto it.
Requirements - Suffficient light for thresholding and clear Cross drawn on
a piece of paper. 
*No multiple edges
*/


//This function takes our converted binary image as input and converts it to\\
denoised image which removes all present holes from the object of interest.
void denoised(cv::Mat& image, cv::Mat& output) {
    cv::Mat inv;
    cv::bitwise_not(image, inv);

    // A image with size greater than the present object is created, it is needed for floodFill()
    cv::Mat mask = cv::Mat::zeros(image.rows + 2, image.cols + 2, CV_8U);

    cv::floodFill(inv, mask, cv::Point(0, 0), 255, 0, cv::Scalar(), cv::Scalar(), 4 + (255 << 8) + cv::FLOODFILL_MASK_ONLY);

    //remove the extra rows/cols added earlier in the initialization of the mask.
    mask(cv::Range(1, mask.rows - 1), cv::Range(1, mask.cols - 1)).copyTo(output);
}


int main()
{   
    //load the pikachu image
    Mat pikachu = imread("pikachu.png", IMREAD_UNCHANGED);
    if (pikachu.empty())
    {
        std::cout << "Cannot load Pikachu image" << std::endl;
        return -1;
    }

    VideoCapture cap(0);
    if (!cap.isOpened())
    {
        std::cout << "Cannot open webcam" << std::endl;
        return -1;
    }

    namedWindow("Video", WINDOW_NORMAL);

    while (true)
    {
        Mat frame;
        cap.read(frame);

        //to find the corners we convert to grayscale, then thresholding and then
        //we find the edges in the thresolded image
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        Mat binary;
        threshold(gray, binary, 100, 255, THRESH_BINARY_INV);

        //cleaning up the binary
        Mat denoised_img;
        denoised(binary, denoised_img);

        //finding the contours (lines)
        std::vector<std::vector<Point>> contours;
        findContours(denoised_img, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        //for checks
        //cout << "contours found" << contours.size() << endl;

        //if 2 edges are found i.e. our cross then perform warping 
        if (1)
        {
            drawContours(frame, contours, 0, Scalar(0, 0, 255), 2);
            drawContours(frame, contours, 1, Scalar(0, 0, 255), 2);

            std::vector<Point2f> srcPoints = { {0.f, 0.f}, {0.f, static_cast<float>(pikachu.rows)},
                                    {static_cast<float>(pikachu.cols), 0.f}, {static_cast<float>(pikachu.cols), static_cast<float>(pikachu.rows)} };
            std::vector<Point2f> dstPoints = { contours[0][0], contours[0][1], contours[1][0], contours[1][1] };
            
            //finding the transformation between the two
            Mat M = getPerspectiveTransform(srcPoints, dstPoints);

            //warping pikachu onto the image
            Mat warpedPikachu;
            warpPerspective(pikachu, warpedPikachu, M, frame.size(), INTER_LINEAR, BORDER_TRANSPARENT);

            //additional blending procedure to ensure our pikachu is well-blended in the frame
            //it masks the background of the pickachu image for smooth blending
            Mat alpha;
            extractChannel(warpedPikachu, alpha, 3);
            Mat invertedAlpha;
            threshold(alpha, invertedAlpha, 0, 255, THRESH_BINARY_INV);
            Mat merged;
            bitwise_and(warpedPikachu, warpedPikachu, merged, invertedAlpha);
            bitwise_or(merged, frame, frame, alpha);
        }
        imshow("Video", frame);
    }
    return 0;
}
