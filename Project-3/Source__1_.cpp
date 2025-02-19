//
//Venkata Sai Advaith Kandiraju
//Assignment 3
//
#include <cmath>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>


using namespace std;
using namespace cv;


//This code initializes the video, and asks the user to perform real time video recognition
// provided the user has stored features for the objects

//Initializing functions before main.

//This converts a grayscale video to binary image
Mat threshold_src(Mat& src);

//This colors the different components in the frame
void color_components(Mat labels, Mat stats, int ncomponents);

//Removes noise/holes from the frame
void binary_filtering(Mat& src, Mat& dst);

//Adds oriented bounding box to the object of interest and adds moment axis
void bounding_box(Mat src, Mat labels, Mat stats, int ncomponents);

//Extracts features for comparison
vector<float> feature_ext(Mat bin_src, int ncomponents, Mat labels, Mat stats);

//Updates excel sheet with new features
int update_database(vector<float> feature_vec);

//Loads the saved excel sheet for image matching later
vector<vector<string>> load_database();

//Updates standard deviation .csv
int update_standard_dev(const vector<vector<string>>& database);

//Loads the standard deviation .csv file for later use
vector<float> load_standard_deviation();

//Calculates distance between vectors for feature matching
float distance(vector<float> A, vector<float> B, vector<float> standard_dev);

//Finds the nearest object using euclidian distance
string find_object(vector<vector<string>>& database, vector<float>& features, vector<float>& standard_dev);

//Finds the nearest object using Knn classifier
string find_object_in_knn(vector<vector<string>>& database, vector<float>& features, vector<float>& standard_dev);

int main(int argc, char* argv[]) {
	Mat src, blur_src, grayscale, binary, binary_filtered, dest;
	VideoCapture* vidsrc;

	Mat frame;
	char method;
	char key;
	vector<vector<string>>database;
	vector<float> standard_dev;

	cout << "\nFor object recognition - 'o'"
		<< " \nFor saving features in database- 'd'"
		<< "\n'q' to exit the video\n";
	cin >> key;

	if (key == 'o') {
		cout << "\nFor euclidian distance method - 'e' "
			<< "\nFor Knn distance method - 'k'\n";
		cin >> method;

		standard_dev = load_standard_deviation();
	}

	database = load_database();

	string capture_name = "Screenshot";

	vidsrc = new VideoCapture(0);
	if (!vidsrc->isOpened()) {
		printf("Unable to open video device\n");
		return(-1);
	}

	Size refS((int)vidsrc->get(CAP_PROP_FRAME_WIDTH),
		(int)vidsrc->get(CAP_PROP_FRAME_HEIGHT));
	printf("Expected size: %d %d\n", refS.width, refS.height);

	namedWindow("Video", 1);

	for (;;) {
		Mat out_frame;
		char quit = waitKey(10);
		*vidsrc >> frame;
		if (frame.empty()) {
			printf("frame is empty\n");
			break;
		}

		GaussianBlur(frame, blur_src, Size(5, 5), BORDER_DEFAULT);

		cvtColor(blur_src, grayscale, COLOR_BGR2GRAY);

		binary = threshold_src(grayscale);

		binary_filtering(binary, binary_filtered);

		Mat stats, labels, centroids;
		Mat inverted;
		bitwise_not(binary, inverted);
		int ncomponents = connectedComponentsWithStats(inverted, labels, stats, centroids, 8, CV_16U);

		vector<float> features = feature_ext(binary_filtered, ncomponents, labels, stats);

		if (key != 'o' and key == 'r') {
			out_frame = frame;
			cout << "key is " << key << endl;
		}
		if (key == 'd') {
			update_database(features);

			vector<vector<string>>database = load_database();

			update_standard_dev(database);

			key = 'r';
		}
		if (key == 'o') {
			string object_labels;
			if (method == 'e') {
				object_labels = find_object(database, features, standard_dev);
			}
			else if (method == 'k') {
				object_labels = find_object_in_knn(database, features, standard_dev);
			}
			else {
				cout << "Wrong key\n";
				return -1;
			}
			Point P(stats.at<int>((int)features[0], CC_STAT_LEFT), stats.at<int>((int)features[0], CC_STAT_TOP));
			putText(frame, object_labels, P, FONT_HERSHEY_PLAIN, 2, Scalar(0, 0, 255));

			out_frame = frame;
		}
		if (quit == 'q') {
			break;
		}
		imshow("Video", frame);
	}
	delete vidsrc;
	
	return 0;
}


Mat threshold_src(Mat& src) {
	int threshold = 128;
	Mat bin_src(src.size(), CV_8U);
	for (int i = 0; i < src.rows; i++) {
		uchar* row_ptr_src = src.ptr<uchar>(i);
		uchar* row_ptr_dst = bin_src.ptr<uchar>(i);
		for (int j = 0; j < src.cols; j++) {
			if (row_ptr_src[j] < threshold) {
				row_ptr_dst[j] = 0;
			}
			else {
				row_ptr_dst[j] = 255;
			}
		}
	}
	return bin_src;
}

void color_components(Mat labels, Mat stats, int ncomponents) {
	Mat segment;
	segment = Mat::zeros(labels.size(), CV_16UC3);
	for (int n = 1; n < ncomponents; n++) {
		if (stats.at<int>(n, CC_STAT_AREA) > 2000) {
			vector<uchar> color(5, (255, 200, 170, 130, 100));
			for (int j = 0; j < segment.rows; j++) {
				for (int k = 0; k < segment.cols; k++) {
					if (labels.at<ushort>(j, k) == n) {
						segment.at<ushort>(j, k, 0) = color[n % 5];
						segment.at<ushort>(j, k, 0) = color[n % 5];
						segment.at<ushort>(j, k, 0) = color[n % 5];
					}
				}
			}
		}
	}
	convertScaleAbs(segment, segment);
	namedWindow("Components", 1);
	imshow("Components", segment);
	waitKey();
}

void binary_filtering(Mat& src, Mat& dst) {
	Mat inverted;
	bitwise_not(src, inverted);
	Mat mask = Mat::zeros(src.rows + 2, src.cols + 2, CV_8U);
	floodFill(inverted, mask, Point(0, 0), 255, 0, Scalar(), Scalar(), 4 + (255 << 8) + FLOODFILL_MASK_ONLY);
	mask(Range(1, mask.rows - 1), Range(1, mask.cols - 1)).copyTo(dst);
}

void bounding_box(Mat src, Mat labels, Mat stats, int ncomponents) {
	for (int n = 1; n < ncomponents; n++) {
		if (stats.at<int>(n, CC_STAT_AREA) > 1000) {
			double n02, n20, n11, al;
			Mat mask = (labels == n);
			Mat query_src;
			vector<Point> Pts;

			cout << "\nArea - " << stats.at<int>(n, CC_STAT_AREA);

			for (int i = 0; i < mask.rows; i++) {
				for (int j = 0; j < mask.cols; j++) {
					if (mask.at<ushort>(i, j) == n) {
						mask.at<ushort>(i, j) = 255;
					}
				}
			}

			convertScaleAbs(mask, query_src);

			Moments moment = moments(mask);

			n11 = moment.nu11;
			n02 = moment.nu02;
			n20 = moment.nu20;

			double val = (2 * n11) / (n20 - n02 + 0.000001);
			al = atan(val) / 2;
			float beta = al + 3.14 / 2;

			Point center(moment.m10 / moment.m00, moment.m01 / moment.m00);

			vector<int> length;
			vector<Point> end_pts;
			cout << "\nCenter is " << center.x << "," << center.y;
			for (int i = 0; i < 2; i++) {
				Point N(center);
				while (1) {
					float s = sin(al + i * (atan(1) * 2));
					float c = cos(al + i * (atan(1) * 2));
					N.x = N.x + 5 * (c);
					N.y = N.y + 5 * (s);
					if ((int)query_src.at<uchar>(N) == 0) {
						break;
					}
				}
				int l = sqrt((N.x - center.x) * (N.x - center.x) + (N.y - center.y) * (N.y - center.y));
				length.push_back(l);
				end_pts.push_back(N);
			}
			Point P1 = end_pts[0];
			Point P2 = 2 * center - P1;
			line(src, P1, P2, Scalar(255, 0, 0), 1, LINE_AA);

			circle(src, center, 2, Scalar(0, 0, 255));

			for (int i = 0; i < query_src.rows; i++) {
				for (int j = 0; j < query_src.cols; j++) {
					if ((int)query_src.at<uchar>(i, j) == 255) {
						Pts.push_back(Point(j, i));
					}
				}
			}

			RotatedRect box = minAreaRect(Pts);
			Point2f vertex[4];
			box.points(vertex);

			for (int i = 0; i < 4; i++) {
				line(src, vertex[i], vertex[(i + 1) % 4], Scalar(0, 255, 0), 1, LINE_AA);
			}

			namedWindow("src", 1);
			imshow("src", src);
			waitKey();
		}
	}
}

vector<float> feature_ext(Mat bin_src, int ncomponents, Mat labels, Mat stats) {
	int area_max = 0; int ind = 100;

	for (int i = 1; i < ncomponents; i++) {
		if (stats.at<int>(i, CC_STAT_AREA) > area_max) {
			area_max = stats.at<int>(i, CC_STAT_AREA);
			ind = i;
		}
	}

	double percent_filled, area, height, breadth;
	Mat mask = (labels == ind);

	Moments moment = moments(mask);
	double hu_moments[7];
	HuMoments(moment, hu_moments);

	area = stats.at<int>(ind, CC_STAT_AREA);
	height = stats.at<int>(ind, CC_STAT_HEIGHT);
	breadth = stats.at<int>(ind, CC_STAT_WIDTH);
	percent_filled = area / (height * breadth);


	vector<float>features;
	vector<float>hu_moms;

	for (int i = 0; i < 6; i++) {
		float k = log10(hu_moments[i]);
		if (isnan(k)) {
			k = 0;
		}
		hu_moms.push_back(k);
	}

	features.push_back(ind); features.push_back(area); features.push_back(height); features.push_back(breadth);
	features.push_back(hu_moms[0]);	features.push_back(hu_moms[1]); features.push_back(hu_moms[2]); features.push_back(hu_moms[3]);
	features.push_back(hu_moms[4]); features.push_back(percent_filled);

	return(features);
}

int update_database(vector<float> feature_vec) {
	char name[50];

	cout << "\nInput label for the object - \n";
	cin >> name;

	ofstream outfile("database.csv", ios_base::app | ios_base::out);
	outfile << name << ",";
	for (float i : feature_vec) {
		outfile << i << ",";
	}
	outfile << "\n";
	outfile.close();

	cout << "database_loaded" << endl;
	return 0;
}

vector<vector<string>> load_database() {
	vector<vector<string>> database;

	ifstream file("database.csv");

	string line;
	while (getline(file, line)) {
		vector<string> record;
		stringstream ss(line);
		string field;

		while (getline(ss, field, ',')) {
			record.push_back(field);
		}

		database.push_back(record);
	}

	file.close();

	return database;
}

int update_standard_dev(const vector<vector<string>>& database) {
	int num_std_dev = database[0].size() - 5;
	vector<float> standard_dev(num_std_dev, 0.0);
	int num_records = database.size();

	vector<double> means(num_std_dev, 0.0);
	for (int j = 5; j < num_std_dev + 5; j++) {
		double sum = 0.0;
		for (int i = 0; i < num_records; i++) {
			double val = stod(database[i][j]);
			sum += val;
		}
		means[j - 5] = sum / num_records;
	}

	vector<double> sum_diff_sq(num_std_dev, 0.0);
	for (int j = 5; j < num_std_dev + 5; j++) {
		for (int i = 0; i < num_records; i++) {
			double val = stod(database[i][j]);
			double diff = val - means[j - 5];
			sum_diff_sq[j - 5] += diff * diff;
		}
	}

	for (int j = 5; j < num_std_dev + 5; j++) {
		double variance = sum_diff_sq[j - 5] / num_records;
		standard_dev[j - 5] = sqrt(variance);
	}

	for (int i = 0; i < standard_dev.size(); i++) {
		if (isnan(standard_dev[i])) {
			standard_dev[i] = 0;
		}
	}

	ofstream file("standard_dev.csv", ios_base::trunc);
	
	if (!file.is_open()) {
		cout << "Not able to open file";
		return -1;
	}

	for (float i : standard_dev) {
		file << i << ",";
	}
	file << "\n";
	file.close();
	return 0;
}

vector<float> load_standard_deviation() {
	vector<float> err;
	ifstream input_file("standard_dev.csv");

	if (!input_file.is_open()) {
		cerr << "Error opening input file!\n";
		return err;
	}

	string line;
	getline(input_file, line);

	stringstream ss(line);
	vector<string> standard_dev_str;

	while (ss.good()) {
		string substr;
		getline(ss, substr, ',');
		standard_dev_str.push_back(substr);
	}

	vector<float> standard_dev(6, 0);

	standard_dev[0] = stod(standard_dev_str[0]);
	standard_dev[1] = stod(standard_dev_str[1]);
	standard_dev[2] = stod(standard_dev_str[2]);
	standard_dev[3] = stod(standard_dev_str[3]);
	standard_dev[4] = stod(standard_dev_str[4]);
	standard_dev[5] = stod(standard_dev_str[5]);
	return standard_dev;
	cout << "Std deviation updated" << endl;
}

float distance(vector<float> A, vector<float> B, vector<float> standard_dev) {
	float dist = 0;
	for (int i = 0; i < A.size() - 1; i++) {
		dist = (A[i] - B[i]) / standard_dev[i];
	}
	return(abs(dist));
}

string find_object(vector<vector<string>>& database, vector<float>& features, vector<float>& standard_dev) {
	int size = database.size();
	int features_n = standard_dev.size();

	vector<vector<float>> database_double(size, vector<float>(features_n));
	vector<string> labels;

	for (int i = 0; i < size; i++) {
		for (int j = 0; j < features_n; j++) {
			database_double[i][j] = stod(database[i][j + 5]);
		}
	}
	for (vector<string> i : database) {
		labels.push_back(i[0]);
	}

	float dist = 0; float min = 999999; int ind = 99999;
	vector<float>features_for_comp(features.begin() + 4, features.end());
	for (int i = 0; i < size; i++) {
		dist = distance(features_for_comp, database_double[i], standard_dev);
		if (dist < min) {
			min = dist;
			ind = i;
		}
	}
	cout << "Similar label is for " << labels[ind] << "\n";
	return labels[ind];
}

string find_object_in_knn(vector<vector<string>>& database, vector<float>& features, vector<float>& standard_dev) {
	int size = database.size();
	int features_n = standard_dev.size();

	vector<vector<float>> database_double(size, vector<float>(features_n));
	vector<string> labels;
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < features_n; j++) {
			database_double[i][j] = stod(database[i][j + 5]);
		}
	}

	for (vector<string> i : database) {
		labels.push_back(i[0]);
	}

	float dist = 0;
	float dist_1 = 0;
	float min = 100000;
	int ind = 1000;
	int count = 1;
	vector<float>features_for_comp(features.begin() + 4, features.end());


	for (int i = 0; i < size; i++) {
		dist = distance(features_for_comp, database_double[i], standard_dev);
		if (i > 0) {

			if (labels[i] == labels[i - 1]) {
				cout << "\nfound same label " << labels[i];
				dist_1 = dist_1 + dist;
				count++;
			}
			if (labels[i] != labels[i - 1] or i == size - 1) {

				dist = dist_1 / count;
				cout << "\nnormalized distance for the label " << labels[i] << " is " << dist;
				dist_1 = 0;
				count = 1;
			}
		}
		if (dist < min) {
			min = dist;
			ind = i;
		}
	}
	cout << "\nCo-relation label found is " << labels[ind] << "\n";
	return labels[ind];
	return 0;
}
