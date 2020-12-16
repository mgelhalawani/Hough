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
int * cudaOptiHough(Mat, uchar **, int *);

struct Coor {
	int x;
	int y;
};

#define THRESHOLD 350
#define MIN_THETA  0
#define MAX_THETA 180

__global__
void basicCUDA(uchar *input, int* output, int rowSize, int colSize, int MAX_D) {

	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
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
void cudaFilter(uchar *input, Coor *output, int *actualSize, int rowSize,
		int colSize) {

	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;

	int index = 0;
	if (row < rowSize && col < colSize) {
		uchar value = *(input + row * colSize + col);

		if (value > 0) {
			Coor data;
			do {
				index++;
				data = {col, row};
				output[index] = data;
			} while (output[index].x != data.x && output[index].y != data.y);
			atomicAdd(actualSize, 1);
		}
	}
}

__global__
void cudaOptiHough(Coor *input, int* output, int size, int MAX_D) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int d = -1;

	if (index < size) {

		Coor point = input[index];

		for (int theta = MIN_THETA; theta <= MAX_THETA; theta += 1) {
			d = point.x * cosf(theta) + point.y * sinf(theta);
			if (d > 0 && d < MAX_D) {
				atomicAdd((output + theta * MAX_THETA + d), 1);
			}
		}
	}
}

int main(int argc, char** argv) {

// Declare the output variables
	Mat cannyImage, stdHostHough, stdDeviceHough, optiDeviceHough;
	const char* default_file = "/home/student/Documents/CUDA/Hough/road8.png";
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
	optiDeviceHough = stdHostHough.clone();

// Standard Hough Line Transform/
	vector<Vec2f> lines; // will hold the results of the detection

	auto time1 = chrono::high_resolution_clock::now();
	lines = cpuHough(cannyImage);
	auto time2 = chrono::high_resolution_clock::now();

	auto duration =
			chrono::duration_cast<chrono::microseconds>(time2 - time1).count();

	cout << "Standard on cpu =  " << duration << "\n";

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

	cout << "Standard on cuda with data copy time = " << duration << "\n";

// Allocate vector output on device
	vector<Vec2f> vecOutput;
	vector<Vec2f>::iterator iterator;

	for (int row = 0; row < MAX_THETA; row++) {
		for (int col = 0; col < cannyImage.cols; col++) {
			if (*(hostOutput + row * cannyImage.cols + col) > THRESHOLD) {
				Vec2f vec = Vec2f(row, col);
				iterator = vecOutput.begin();
				vecOutput.insert(iterator, vec);
			}
		}
	}

	free(hostOutput);
	drawLines2(stdDeviceHough, vecOutput);

	int *optiSize = (int *) malloc(sizeof(int));

	time1 = chrono::high_resolution_clock::now();
	int *cudaOptiOutput = cudaOptiHough(cannyImage, rawData, optiSize);
	time2 = chrono::high_resolution_clock::now();

	duration =
			chrono::duration_cast<chrono::microseconds>(time2 - time1).count();

	cout << "Optimized on cuda with data copy time  = " << duration << "\n";

	// Allocate vector output on device
	vector<Vec2f> vecOutput2;

	for (int row = 0; row < *optiSize; row++) {
		for (int col = 0; col < MAX_THETA; col++) {
			if (*(cudaOptiOutput + row * MAX_THETA + col) > THRESHOLD) {
				Vec2f vec = Vec2f(row, col);
				iterator = vecOutput2.begin();
				vecOutput2.insert(iterator, vec);
			}
		}
	}

	drawLines2(optiDeviceHough, vecOutput2);

// Show results
	imwrite("./Source.png", srcImage);
	imwrite("./Standard-Hough.png", stdHostHough);
	imwrite("./Device-Standard-Hough.png", stdDeviceHough);
	imwrite("./Device-Opti-Hough.png", stdDeviceHough);

	free(rawData);
	return 0;
}

vector<Vec2f> cpuHough(Mat cannyImage) {
	vector<Vec2f> lines;
	HoughLines(cannyImage, lines, 1, CV_PI / 180, THRESHOLD, 0, 0); // runs the actual detection
	return lines;
}

int* cudaBasicHough(Mat cannyImage, uchar **rawData) {
// Allocate input matrix on device
	uchar *deviceInput;
	int size = cannyImage.rows * cannyImage.cols * sizeof(uchar);

	cudaMalloc((void **) &deviceInput, size);


// Copy image to device
	cudaMemcpy(deviceInput, rawData[0], size, cudaMemcpyHostToDevice);

// Allocate output matrix on device
	int *deviceOutput;
	cudaMalloc(&deviceOutput, size);
	cudaMemset(deviceOutput, 0, size);

	int MAX_D = sqrt(pow(cannyImage.rows, 2) + pow(cannyImage.cols, 2));

	cout << "rows: " << cannyImage.rows << ", cols: " << cannyImage.cols << "\n";

// Define block and grid size
	dim3 blockSize(32, 32);
	int gridRows = (cannyImage.rows + blockSize.x - 1) / blockSize.x;
	int gridCols = (cannyImage.cols + blockSize.y - 1) / blockSize.y;
	dim3 gridSize = dim3(gridRows, gridCols);

	auto time1 = chrono::high_resolution_clock::now();

	basicCUDA<<<gridSize, blockSize>>>(deviceInput, deviceOutput,
			cannyImage.rows, cannyImage.cols, MAX_D);

	cudaDeviceSynchronize();

	auto time2 = chrono::high_resolution_clock::now();

	auto duration =
			chrono::duration_cast<chrono::microseconds>(time2 - time1).count();

	cout << "Standard on cuda without data copy time =  " << duration << "\n";

	int *hostOutput = (int *) calloc(cannyImage.rows * cannyImage.cols,
			sizeof(int));
	// Copy image from device
	cudaMemcpy(hostOutput, deviceOutput, size, cudaMemcpyDeviceToHost);
	cudaFree(deviceInput);
	cudaFree(deviceOutput);
	return hostOutput;
}

int* cudaOptiHough(Mat cannyImage, uchar **rawData, int *houghSize) {
	int size = cannyImage.rows * cannyImage.cols * sizeof(uchar);

	uchar *deviceInput;

	cudaMalloc((void **) &deviceInput, size);

	// Copy image to device
	cudaMemcpy(deviceInput, rawData[0], size, cudaMemcpyHostToDevice);

	// Allocate output matrix on device
	Coor *pointsList;
	cudaMalloc(&pointsList, size);

	int *deviceSize;
	cudaMalloc((void **) &deviceSize, sizeof(int));
	cudaMemset(deviceSize, 0, sizeof(int));

// Define block and grid size
	dim3 blockSize(32, 32);
	int gridRows = (cannyImage.rows + blockSize.x - 1) / blockSize.x;
	int gridCols = (cannyImage.cols + blockSize.y - 1) / blockSize.y;
	dim3 gridSize = dim3(gridRows, gridCols);

	int MAX_D = sqrt(pow(cannyImage.rows, 2) + pow(cannyImage.cols, 2));

	auto time1 = chrono::high_resolution_clock::now();

	cudaFilter<<<gridSize, blockSize>>>(deviceInput, pointsList, deviceSize,
			cannyImage.rows, cannyImage.cols);

	cudaDeviceSynchronize();

	auto time11 = chrono::high_resolution_clock::now();

	int dataSize;
	cudaMemcpy(&dataSize, deviceSize, sizeof(int), cudaMemcpyDeviceToHost);

	int *deviceHoughOutput;
	cudaMalloc(&deviceHoughOutput, dataSize * MAX_THETA * sizeof(int));
	cudaMemset(deviceHoughOutput, 0, dataSize * MAX_THETA * sizeof(int));

	auto time21 = chrono::high_resolution_clock::now();
	cudaOptiHough<<<gridSize, blockSize>>>(pointsList, deviceHoughOutput,
			dataSize, MAX_D);

	cudaDeviceSynchronize();
	auto time2 = chrono::high_resolution_clock::now();

	auto duration =
					chrono::duration_cast<chrono::microseconds>(time11 - time1).count();
	cout << "filter step on =  " << duration << "\n";

	duration =
					chrono::duration_cast<chrono::microseconds>(time2 - time21).count();

	cout << "Hough Step on cuda =  " << duration << "\n";

	duration =
			chrono::duration_cast<chrono::microseconds>(time2 - time1).count();

	cout << "Optimized on cuda without data copy time =  " << duration << "\n";

	*houghSize = dataSize;

	int *hostOutput = (int *) calloc(dataSize * MAX_THETA, sizeof(int));
	cudaMemcpy(hostOutput, deviceHoughOutput, dataSize * MAX_THETA * sizeof(int),
			cudaMemcpyDeviceToHost);

	cudaFree(pointsList);
	cudaFree(deviceHoughOutput);
	cudaFree(deviceInput);

	return hostOutput;
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
