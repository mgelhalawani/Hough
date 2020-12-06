/*
 ============================================================================
 Name        : Hough.cu
 Author      : 
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <cuda_fp16.h>
#include <map>

#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

void drawLines(Mat, vector<Vec2f>);
void drawLines2(Mat, vector<Vec2f>);
void drawCircles(Mat, int*, int, int);

vector<Vec2f> cpuHough(Mat);
int * cudaBasicHough(Mat, uchar **);
void cudaOptiHough(Mat, uchar **);

struct coor {
	int x;
	int y;
};

__global__
void basicCUDA(uchar *input, int* output, int rowSize, int colSize, int MAX_D) {

	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;

	int MIN_THETA = 0;
	int MAX_THETA = 180;
	int d = -1;

	if (row < rowSize && col < colSize) {
		uchar value = *(input + row * colSize + col);
		if (value > 0) {
			for (int theta = MIN_THETA; theta <= MAX_THETA; theta += 1) {
				d = col * cosf(theta) + row * sinf(theta);
				if (d > 0 && d < MAX_D) {
					atomicAdd((output + theta * colSize + d), 1);
				}
			}
		}
	}
}

__global__
void cudaFilter(uchar *input, coor *output, int *actualSize,
		int rowSize, int colSize) {

	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;

	int index = 0;
	if (row < rowSize && col < colSize) {
		uchar value = *(input + row * colSize + col);

		if (value > 0) {
			coor data;
			do {
				index++;
				data = {col, row};
				output[index] = data;
			} while (output[index].x != data.x && output[index].y != data.y);
			atomicAdd(actualSize,1);
		}
	}
}

__global__
void cudaOptiHough (){

}

int main(int argc, char** argv) {

// Declare the output variables
	Mat cannyImage, stdHostHough, stdDeviceHough;
	const char* default_file = "/home/student/Documents/CUDA/Hough/road2.png";
	const char* filename = argc >= 2 ? argv[1] : default_file;

// Loads an image
	Mat srcImage = imread(samples::findFile(filename), IMREAD_GRAYSCALE);

// Check if image is loaded fine
	if (srcImage.empty()) {
		printf(" Error opening image\n");
		printf(" Program Arguments: [image_name -- default %s] \n",
				default_file);
		return -1;
	}

// Edge detection
	Canny(srcImage, cannyImage, 50, 200, 3);
	imwrite("./canny.png", cannyImage);

// convert back the image to RGB so that we can draw coloured lines
	cvtColor(cannyImage, stdHostHough, COLOR_GRAY2BGR);
	stdDeviceHough = stdHostHough.clone();

// Standard Hough Line Transform/
	vector<Vec2f> lines; // will hold the results of the detection

	auto time1 = chrono::high_resolution_clock::now();
	lines = cpuHough(cannyImage);
	auto time2 = chrono::high_resolution_clock::now();

	auto duration =
			chrono::duration_cast<chrono::microseconds>(time2 - time1).count();

	cout << "cpu =  " << duration << "\n";

// Draw the lines
	drawLines(stdHostHough, lines);

// Extract data from Mat
	uchar **rawData = new uchar*[cannyImage.rows];
	rawData[0] = new uchar[cannyImage.rows * cannyImage.cols];

	for (int index = 1; index < cannyImage.rows; ++index) {
		rawData[index] = rawData[index - 1] + cannyImage.cols;
	}

	for (int row = 0; row < cannyImage.rows; ++row) {
		for (int col = 0; col < cannyImage.cols; ++col) {
			uchar value = cannyImage.at<uchar>(row, col);
			rawData[row][col] = value;
		}
	}

	time1 = chrono::high_resolution_clock::now();

	int *hostOutput = cudaBasicHough(cannyImage, rawData);

	time2 = chrono::high_resolution_clock::now();

	duration =
			chrono::duration_cast<chrono::microseconds>(time2 - time1).count();

	cout << "cuda = " << duration << "\n";

// Allocate vector output on device
	vector<Vec2f> vecOutput;
	vector<Vec2f>::iterator iterator;

	for (int row = 0; row < 180; row++) {
		for (int col = 0; col < 1280; col++) {
			if (*(hostOutput + row * cannyImage.cols + col) > 350) {
				Vec2f vec = Vec2f(row, col);
				iterator = vecOutput.begin();
				vecOutput.insert(iterator, vec);
			}
		}
	}

	drawLines2(stdDeviceHough, vecOutput);

	cudaOptiHough(cannyImage, rawData);

// Show results
	imwrite("./Source.png", srcImage);
	imwrite("./Standard-Hough.png", stdHostHough);
	imwrite("./Device-Standard-Hough.png", stdDeviceHough);

	return 0;
}

vector<Vec2f> cpuHough(Mat cannyImage) {
	vector<Vec2f> lines;
	HoughLines(cannyImage, lines, 1, CV_PI / 180, 350, 0, 0); // runs the actual detection
	return lines;
}

int* cudaBasicHough(Mat cannyImage, uchar **rawData) {
// Allocate input matrix on device
	uchar *deviceInput;
	int size = cannyImage.rows * cannyImage.cols * sizeof(int);
	cudaMalloc((void **) &deviceInput, size);

// Copy image to device
	cudaMemcpy(deviceInput, rawData[0], size, cudaMemcpyHostToDevice);

// Allocate output matrix on device
	int *deviceOutput;
	cudaMalloc(&deviceOutput, size);
	cudaMemset(deviceOutput, 0, size);

	int MAX_D = sqrt(pow(cannyImage.rows, 2) + pow(cannyImage.cols, 2));

// Define block and grid size
	dim3 blockSize(32, 32);
	int gridRows = (cannyImage.rows + blockSize.x - 1) / blockSize.x;
	int gridCols = (cannyImage.cols + blockSize.y - 1) / blockSize.y;
	dim3 gridSize = dim3(gridRows, gridCols);

	basicCUDA<<<gridSize, blockSize>>>(deviceInput, deviceOutput,
			cannyImage.rows, cannyImage.cols, MAX_D);

	cudaDeviceSynchronize();

	int *hostOutput = (int *) calloc(cannyImage.rows * cannyImage.cols,
			sizeof(int));
// Copy image to device
	cudaMemcpy(hostOutput, deviceOutput, size, cudaMemcpyDeviceToHost);
	cudaFree(deviceInput);
	cudaFree(deviceOutput);
	return hostOutput;
}

void cudaOptiHough(Mat cannyImage, uchar **rawData) {
	int size = cannyImage.rows * cannyImage.cols * sizeof(coor);

	uchar *deviceInput;
	cudaMalloc((void **) &deviceInput, size);

	int *deviceSize;
	cudaMalloc((void **) &deviceSize, sizeof(int));
	cudaMemset(deviceSize, 0, sizeof(int));

	// Copy image to device
	cudaMemcpy(deviceInput, rawData[0], size, cudaMemcpyHostToDevice);

// Allocate output matrix on device
	coor *deviceOutput;
	cudaMalloc(&deviceOutput, size);

// Define block and grid size
	dim3 blockSize(32, 32);
	int gridRows = (cannyImage.rows + blockSize.x - 1) / blockSize.x;
	int gridCols = (cannyImage.cols + blockSize.y - 1) / blockSize.y;
	dim3 gridSize = dim3(gridRows, gridCols);

	cudaFilter<<<gridSize, blockSize>>>(deviceInput, deviceOutput,
			deviceSize, cannyImage.rows, cannyImage.cols);

	cudaDeviceSynchronize();

	coor *hostOutput = (coor *) malloc(size);

// Copy image to device
	cudaMemcpy(hostOutput, deviceOutput, size, cudaMemcpyDeviceToHost);
	cudaFree(deviceInput);
	cudaFree(deviceOutput);

	int *dataSize = (int *) malloc(sizeof(int));
	cudaMemcpy(dataSize, deviceSize, sizeof(int), cudaMemcpyDeviceToHost);

	cout << "dataSize = " << *dataSize;

//	for (int i = 0; i < cannyImage.rows * cannyImage.cols; i++) {
//		if (hostOutput[i].x > 877) {
//			cout << "x = " << hostOutput[i].x << ", y = " << hostOutput[i].y
//					<< "\n";
//		}
//	}
	free(hostOutput);
	free(dataSize);

}

void drawLines(Mat cdst, vector<Vec2f> lines) {

	for (size_t i = 0; i < lines.size(); i++) {
		float rho = lines[i][0], theta = lines[i][1];
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a * rho, y0 = b * rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		line(cdst, pt1, pt2, Scalar(0, 0, 255), 3, LINE_AA);
	}
}

void drawLines2(Mat cdst, vector<Vec2f> lines) {

	for (size_t i = 0; i < lines.size(); i++) {
		float rho = lines[i][1], theta = lines[i][0];
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a * rho, y0 = b * rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		line(cdst, pt1, pt2, Scalar(0, 0, 255), 3, LINE_AA);
	}
}

void drawCircles(Mat cdst, int* points, int rowSize, int colSize) {

	for (int row = 0; row < rowSize; row++) {
		for (int col = 0; col < colSize; col++) {
			if (*(points + row * colSize + col) == 1) {
				Point center;
				center.x = col;
				center.y = row;
				circle(cdst, center, 0, Scalar(0, 0, 255), 2);
			}
		}
	}
}
