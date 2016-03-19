/**
Team members: Yizhi Huang, Yue Zhou, Yingqiao Xiong, Annalisa Chen



*/

#include "stdafx.h"
//opencv libraries
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
//C++ standard libraries
#include <iostream>
#include <vector>
#include <algorithm>

using namespace cv;
using namespace std;

//function declarations

/**
Function that returns the maximum of 3 integers
@param a first integer
@param b second integer
@param c third integer
*/
int myMax(int a, int b, int c);

/**
Function that returns the minimum of 3 integers
@param a first integer
@param b second integer
@param c third integer
*/
int myMin(int a, int b, int c);

/**
Function that detects whether a pixel belongs to the skin based on RGB values
@param src The source color image
@param dst The destination grayscale image where skin pixels are colored white and the rest are colored black
*/
void mySkinDetect(Mat& src, Mat& dst);

/**
Function that does frame differencing between the current frame and the previous frame
@param src The current color image
@param prev The previous color image
@param dst The destination grayscale image where pixels are colored white if the corresponding pixel intensities in the current
and previous image are not the same
*/
void myFrameDifferencing(Mat& prev, Mat& curr, Mat& dst);

/**
Function that accumulates the frame differences for a certain number of pairs of frames
@param mh Vector of frame difference images
@param dst The destination grayscale image to store the accumulation of the frame difference images
*/
void myMotionEnergy(vector<Mat> mh, Mat& dst);

//void vertical(Mat& src, vector<int> mh);

void myTintImage(Mat& src, Mat& dst, int channel);

int main()
{

	//----------------
	//a) Reading a stream of images from a webcamera, and displaying the video
	//----------------
	// For more information on reading and writing video: http://docs.opencv.org/modules/highgui/doc/reading_and_writing_images_and_video.html
	// open the video camera no. 0
	VideoCapture cap(0);

	// if not successful, exit program
	if (!cap.isOpened())
	{
		cout << "Cannot open the video cam" << endl;
		return -1;
	}

	Mat frame0;

	// read a new frame from video
	bool bSuccess0 = cap.read(frame0);

	//if not successful, break loop
	if (!bSuccess0)
	{
		cout << "Cannot read a frame from video stream" << endl;
	}

	//create a window called "MyVideoFrame0"
	namedWindow("MyVideo0", WINDOW_AUTOSIZE);
	imshow("MyVideo0", frame0);

	//create a window called "MyVideo"
	namedWindow("MyVideo", WINDOW_AUTOSIZE);
	namedWindow("MyVideoDF", WINDOW_AUTOSIZE);
	namedWindow("MyVideoMH", WINDOW_AUTOSIZE);
	namedWindow("Skin", WINDOW_AUTOSIZE);


	vector<Mat> motionHistory;
	Mat mh1;
	Mat mh2;
	Mat mh3;
	mh1 = Mat::zeros(frame0.rows, frame0.cols, CV_8UC1);
	mh2 = mh1.clone();
	mh3 = mh1.clone();
	motionHistory.push_back(mh1);
	motionHistory.push_back(mh2);
	motionHistory.push_back(mh3);


	while (1)
	{
		// read a new frame from video
		Mat frame;
		bool bSuccess = cap.read(frame);

		//if not successful, break loop
		if (!bSuccess)
		{
			cout << "Cannot read a frame from video stream" << endl;
			break;
		}

		imshow("MyVideo", frame);
		
		// destination frame
		Mat frameDest;
		frameDest = Mat::zeros(frame.rows, frame.cols, CV_8UC1); 

		// blurring
		Mat blurDest = Mat::zeros(frame.rows, frame.cols, CV_8UC1);
		blur(frame, blurDest, Size(3, 3));

		// skin color detection
		Mat skinDest = Mat::zeros(frame.rows, frame.cols, CV_8UC1);
		mySkinDetect(blurDest, skinDest);
		imshow("Skin", skinDest);

		// background differencing
		Mat diffDest = Mat::zeros(frame0.rows, frame0.cols, CV_8UC1);
		// call myFrameDifferencing function
		myFrameDifferencing(frame0, frame, diffDest);
		imshow("MyVideoDF", diffDest);
		
		// visualizing motion history
		motionHistory.erase(motionHistory.begin());
		motionHistory.push_back(diffDest);
		Mat myMH = Mat::zeros(frame0.rows, frame0.cols, CV_8UC1);
		// call myMotionEnergy function
		myMotionEnergy(motionHistory, myMH);
		imshow("MyVideoMH", myMH); 


		// find contour

		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;

		// Documentation for finding contours: http://docs.opencv.org/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=findcontours#findcontours
		findContours(skinDest, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
		cout << "The number of contours detected is: " << contours.size() << endl;

		vector<vector<int> > hullsI(contours.size());
		vector<vector<Point> > hullsP(contours.size());
		vector<vector<Vec4i> > defects(contours.size());

		for (int i = 0; i <contours.size(); ++i){
			//find hulls
			convexHull(Mat(contours[i]), hullsI[i], false);
			convexHull(Mat(contours[i]), hullsP[i], false);
			//find defects  
			convexityDefects(contours[i], hullsI[i], defects[i]);
		}

		Mat contour_output = Mat::zeros(skinDest.size(), CV_8UC3);
		// Find largest contour
		int maxsize = 0;
		int maxind = 0;
		Rect boundrec;
		for (int i = 0; i < contours.size(); i++)
		{
			// Documentation on contourArea: http://docs.opencv.org/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#
			double area = contourArea(contours[i]);
			if (area > maxsize) {
				maxsize = (int)area;
				maxind = i;
				boundrec = boundingRect(contours[i]);
			}
		}

		Mat frame1 = frame;
		drawContours(contour_output, contours, maxind, Scalar(255, 0, 0), CV_FILLED, 8, hierarchy);
		drawContours(contour_output, contours, maxind, Scalar(0, 0, 255), 2, 8, hierarchy);
		drawContours(frame1, contours, maxind, Scalar(0, 0, 255), 2, 8, hierarchy);
		rectangle(contour_output, boundrec, Scalar(0, 255, 0), 1, 8, 0);
		rectangle(frame1, boundrec, Scalar(255, 0, 0), 1, 8, 0);
		namedWindow("Contours", CV_WINDOW_AUTOSIZE);
		imshow("Contours", contour_output);

		if (contours.size() > 0)
		{
			drawContours(frame1, contours, maxind, Scalar(0, 0, 255), 2, 8, hierarchy);
			drawContours(frame1, hullsP, maxind, Scalar(0, 255, 0), 2, 8, hierarchy);
		}
		
		// Fingle Detection
		int countFingers = 0;
		Point2f handCenter;
		float radius;
		vector<Vec4i> maxDefects = defects[maxind];
		vector<Point> maxContour = contours[maxind];
		//output the center of hand
		minEnclosingCircle(maxContour, handCenter, radius);
		circle(frame1, handCenter, 10, Scalar(0, 0, 255), 2, 8);

		//analyze defects in the hand: 
		//get the position of finger tips and depth 
		for (int i = 0; i < maxDefects.size(); i++){
			int startIndx = maxDefects[i].val[0]; 
			Point fingerTip(maxContour[startIndx]);
		    int depth = (int) (maxDefects[i].val[3]) / 256;
			
			//compare depth and the position of fingertips and hand center
			//count fingers
			if (depth > 8 && fingerTip.y < handCenter.y)
			{
				circle(frame1, fingerTip, 3, Scalar(0, 125, 200), 4);
				countFingers++;
			}
		}
		
		//output graphical display on the screen
		if (countFingers == 0){
			putText(frame1, "Fist", Point(50, 100), 2, 4, Scalar(50, 0, 125), 3, 8);
		}
		else
		{
			if (countFingers >= 5){
				putText(frame1, "High Five!", Point(50, 100), 2, 3, Scalar(50, 0, 125), 3, 8);

			}
			else
			{
				putText(frame1, to_string(countFingers) + "", Point(50, 100), 2, 4, Scalar(50, 0, 125), 3, 8);
			}
		}

		namedWindow("fingerVideo", CV_WINDOW_AUTOSIZE);
		imshow("fingerVideo", frame1);
		


		// 0: BLUE channel, 1: GREEN channel, 2: RED channel
		// when hand moves vertically, the screen turns to green
		// when hand moves horizontally, the screen turns to red

		Mat tint_frame = frame;

		// horizontal histogram
		for (int x = 0; x< myMH.rows; x++){
			int i = 0;
			for (int y = 0; y< myMH.cols; y++) {
				if (myMH.at<uchar>(x,y) == 255)
					i += 1;
			}
			if (i > 350) {
				myTintImage(frame, tint_frame, 2);
				break;
			}
		}
		
		
		// vectical histogram
		for (int x = 0; x< myMH.cols; x++){
			int i = 0;
			for (int y = 0; y< myMH.rows; y++) {
				if (myMH.at<uchar>(y,x) == 255)
					i += 1;
			}
			if (i > 350) {
				myTintImage(frame, tint_frame, 1);
				break;
			}
		}
		
		namedWindow("colorVideo", CV_WINDOW_AUTOSIZE);
		imshow("colorVideo", tint_frame);

		// reset frame
		imshow("MyVideo0", frame0);
		frame0 = frame;

		// wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
		if (waitKey(30) == 27)
		{
			cout << "esc key is pressed by user" << endl;
			break;
		}

	}
	cap.release();
	return 0;
}

//Function that returns the maximum of 3 integers
int myMax(int a, int b, int c) {
	int m = a;
	(void)((m < b) && (m = b));
	(void)((m < c) && (m = c));
	return m;
}

//Function that returns the minimum of 3 integers
int myMin(int a, int b, int c) {
	int m = a;
	(void)((m > b) && (m = b));
	(void)((m > c) && (m = c));
	return m;
}

//Function that detects whether a pixel belongs to the skin based on RGB values
void mySkinDetect(Mat& src, Mat& dst) {
	//Surveys of skin color modeling and detection techniques:
	//Vezhnevets, Vladimir, Vassili Sazonov, and Alla Andreeva. "A survey on pixel-based skin color detection techniques." Proc. Graphicon. Vol. 3. 2003.
	//Kakumanu, Praveen, Sokratis Makrogiannis, and Nikolaos Bourbakis. "A survey of skin-color modeling and detection methods." Pattern recognition 40.3 (2007): 1106-1122.
	for (int i = 0; i < src.rows; i++){
		for (int j = 0; j < src.cols; j++){
			//For each pixel, compute the average intensity of the 3 color channels
			Vec3b intensity = src.at<Vec3b>(i, j); //Vec3b is a vector of 3 uchar (unsigned character)
			int B = intensity[0]; int G = intensity[1]; int R = intensity[2];
			if ((R > 95 && G > 40 && B > 20) && (myMax(R, G, B) - myMin(R, G, B) > 15) && (abs(R - G) > 15) && (R > G) && (R > B)){
				dst.at<uchar>(i, j) = 255;
			}
		}
	}
}

//Function that does frame differencing between the current frame and the previous frame
void myFrameDifferencing(Mat& prev, Mat& curr, Mat& dst) {
	//For more information on operation with arrays: http://docs.opencv.org/modules/core/doc/operations_on_arrays.html
	//For more information on how to use background subtraction methods: http://docs.opencv.org/trunk/doc/tutorials/video/background_subtraction/background_subtraction.html
	absdiff(prev, curr, dst);
	Mat gs = dst.clone();
	cvtColor(dst, gs, CV_BGR2GRAY);
	dst = gs > 50;
	Vec3b intensity = dst.at<Vec3b>(100, 100);
}

//Function that accumulates the frame differences for a certain number of pairs of frames
void myMotionEnergy(vector<Mat> mh, Mat& dst) {
	Mat mh0 = mh[0];
	Mat mh1 = mh[1];
	Mat mh2 = mh[2];

	for (int i = 0; i < dst.rows; i++){
		for (int j = 0; j < dst.cols; j++){

			if (mh0.at<uchar>(i, j) == 255 || mh1.at<uchar>(i, j) == 255 || mh2.at<uchar>(i, j) == 255){
				dst.at<uchar>(i, j) = 255;
			}
		}
	}
}

//Creates a tinted image from a color image.
void myTintImage(Mat& src, Mat& dst, int channel)
{
	dst = src.clone(); //the clone methods creates a deep copy of the matrix
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			//For each pixel, suppress the channels other than that passed in the argument of the function
			dst.at<Vec3b>(i, j)[(channel + 1) % 3] = 0;
			dst.at<Vec3b>(i, j)[(channel + 2) % 3] = 0;
		}
	}
}
