/*
  Creators - Shreejit Deshmukh, Venkata Advaith Kandiraju
  Pattern recognition and Computer vision Sping 23
  Content based image ranking
*/

#include <dirent.h>
#include <matching_func.h>
#include <opencv2/opencv.hpp>
#include <tuple>
#include <Windows.h>

//Defining global (float,string) vector for storing metric associated with given image
typedef std::vector< std::tuple<float, std::string> > tuple_list;

/************************************************************************************************************
    The following program performs content based image retrieval for a target image. The input is just the 
    directory for target and query image. Then by following the guided instructions, we input the search method
    and target image name if needed. Then after sorting (Method 1 (single program)) we input number of ranked 
    images we want to see. It will also generate a ranked collage for visualization as a mini extension.
 ************************************************************************************************************/

int main(int argc, char* argv[]) {
    char dirname[256];              //directory name
    char current[256];              //for storing the name of current image while looping
    char target[256];               //query image
    char key;                       //for selection of query
    char target_path[256];
    float dist;                     //similarity metric for target and query
    std::vector<cv::Mat> images;    //for displaying combined images
    tuple_list metric_list;         //for storing image_name and metric as array        
    
    //For looping through each image
    FILE* fp;
    DIR* dirp;
    struct dirent* dp;
    int i,N, flag = 0;
    cv::Mat T, C;

    // check for sufficient arguments
    if (argc < 2) {
        std::cout << "usage: %s <directory path>\n" << argv[0] << std::endl;
        exit(-1);
    }

    // get the directory path
    strcpy_s(dirname, argv[1]);
    std::cout << "\nProcessing directory %s\t" << dirname;

    // open the directory
    dirp = opendir(dirname);
    if (dirp == NULL) {
        std::cout << "\nCannot open directory %s\n" << dirname;
        exit(-1);
    }
    
    //instructions
    std::cout << "\n\nSpecify the function with key to perform the following task -\n\
'b' for baseline matching\n\
'h' for histogram matching\n\
'm' for multi-histogram matching\n\
't' for texture and color matching\n\
'c' to go to custom design\n\
'e' to go to extension mode\n- ";
    
    //key input from user
    std::cin >> key;
    
    //Custom design and extension does not require target image
    if (key != 'c' and key != 'e') {
        std::cout << "\nSpecify the target image for comparison - ";
        std::cin >> target;

        //target path for later image read
        strcpy_s(target_path, dirname);
        strcat_s(target_path, "/");
        strcat_s(target_path, target);

        //generate histogram of target image once
        T = cv::imread(target_path);
        if (T.empty()) {
            std::cout << "Image not found";
            std::cin.get();
            return -1;
        }
    }

    
    // loop over all the files in the image file listing
    while ((dp = readdir(dirp)) != NULL) {

        // check if the file is an image
        if (strstr(dp->d_name, ".jpg") ||
            strstr(dp->d_name, ".png") ||
            strstr(dp->d_name, ".ppm") ||
            strstr(dp->d_name, ".tif")) {

            //current image path
            strcpy_s(current, dirname);
            strcat_s(current, "/");
            strcat_s(current, dp->d_name);

            C = cv::imread(current);                //no need for check since if target image is non empty current can't be non-empty

            switch (key) {
            case 'b':                               //for base line matching
                dist = base_matching(T, C);
                break;
            case 'h':                               //for full image histogram matching
                dist = histogram_matching(T, C);
                break;
            case 'm':                               //for multi-histogram matching
                dist = multi_hist(T, C);
                break;
            case 't':
                dist = text_color(T, C);            //for texture-color fusion matching
                break;
            case 'c':                               //custom design for extracting images with blue sky
                dist = custom_func(C);
                break;
            case 'e':
                dist = banana_finder(C);            //custom desing for finding banana
                break;
             //wrong key - checker
            default:
                std::cout << "invalid_key" << std::endl;
                std::cin.get();
                return -1;
                break;
            }
 
            
            //Creating array of image comparison metric and the image name
            metric_list.push_back(std::tuple<float, std::string>(dist, current));
        }
    }
    
    std::cout << "\n<< Sorting Done >>\n\n\
Please enter the number of co-related images in ascending order you want to print ";
    
    //Number of ranked queries to print
    std::cin >> N;

    //Finding number of elements in our query/array
    int size_met = metric_list.size();
    std::sort(metric_list.begin(), metric_list.end());                         
   
    //For L2 norm metric -> Lesser distance -> greater corelation || for histogram intersection we use greater distance -> greater corelation

    //printing queries in order
    for (int i = 1; i <= N + 1; i++) {
        
        //k decides if to print in ascending or descending order depending on the design metric
        
        int k;
        if (key == 'c' or key == 'e') {
            k = size_met - i;
        }
        else {
            k = i - 1;
        }
        //for not counting target/reference image in ranking
        if (i != 1) {
            std::cout << "The " << i - 1 << " co-relation ranked image is " << std::get<1>(metric_list[k]) << " with metric " << std::get<0>(metric_list[k]) << std::endl;
        }
        //for collage creating vector of images
        cv::Mat image = cv::imread(std::get<1>(metric_list[k]));
        images.push_back(image);
    }
 
    int collageWidth = images[0].cols;
    int collageHeight = 0;
    
    for (int i = 0; i < images.size(); i++) {
        collageHeight += images[i].rows;
    }

    // Create a black image with the desired size
    cv::Mat collage = cv::Mat::zeros(collageHeight, collageWidth, CV_8UC3);

    int y = 0;
    for (int i = 0; i < images.size(); i++) {
        cv::Mat roi(collage, cv::Rect(0, y, collageWidth, images[i].rows));
        images[i].copyTo(roi);
        y += images[i].rows;
    }

    // Show the collage
    cv::imwrite("Collage.jpg", collage);
    cv::waitKey(0);

    printf("\nTerminating\n");
    return(0);
}


