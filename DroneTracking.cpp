// DroneTracking.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/core.hpp>

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay(Mat frame);

/** Global variables */
String drone_cascade_name = "cascade2.xml";
CascadeClassifier drone_cascade;
string window_name = "Capture - Drone detection";
RNG rng(12345);

/** @function main */
int main(int argc, const char** argv)
{
	
	//process video
	VideoCapture capture;
	Mat frame;

	//-- 1. Load the cascades
	if (!drone_cascade.load(drone_cascade_name)) { printf("--(!)Error loading\n"); return -1; };

	/*
	//-- 2. Read the video stream
	capture.open("video2.mov");

	if (capture.isOpened())
	{
		while (true)
		{
			capture >> frame;

			//-- 3. Apply the classifier to the frame
			if (!frame.empty())
			{
				detectAndDisplay(frame);
			}
			else
			{
				printf(" --(!) No captured frame -- Break!"); break;
			}

			int c = waitKey(10);
			if ((char)c == 'c') { break; }
		}
	}
	*/

	//process image
	Mat image;
	image = imread("IMG_2352-1.png", CV_LOAD_IMAGE_COLOR);   // Read the file

	if (!image.data)                              // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	detectAndDisplay(image);

	//namedWindow("Display window", WINDOW_AUTOSIZE);// Create a window for display.
	//imshow("Display window", image);                   // Show our image inside it.

	waitKey(0);                                          // Wait for a keystroke in the window
	

	return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay(Mat frame)
{
	std::vector<Rect> drones;
	Mat frame_gray;

	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	//-- Detect drones
	drone_cascade.detectMultiScale(frame_gray, drones, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

	for (size_t i = 0; i < drones.size(); i++)
	{
		Point center(drones[i].x + drones[i].width*0.5, drones[i].y + drones[i].height*0.5);
		ellipse(frame, center, Size(drones[i].width*0.5, drones[i].height*0.5), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);

		Mat droneROI = frame_gray(drones[i]);
	}
	//-- Show what you got
	imshow(window_name, frame);
}

