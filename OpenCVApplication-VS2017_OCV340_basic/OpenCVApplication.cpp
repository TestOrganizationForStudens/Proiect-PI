// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"

float Average(Mat src);
float Deviation(Mat src, float avr);

void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, CV_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = 255 - val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the ?diblook style?
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int) src.step; // no dword alignment is done !!!
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Componentele d eculoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;		// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int) k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = cvWaitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == CV_EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
		}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

/* Histogram display function - display a histogram using bars (simlilar to L3 / PI)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i<hist_cols; i++)
	if (hist[i] > max_hist)
		max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

/********************************************************** BIT IMAGE *******************************************************************/

void bitImage() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat bitImg = src.clone();
		int height = src.rows;
		int width = src.cols;
		float average = Average(src);
		float deviation = Deviation(src, average);
		int k1 = 1, k2 = -1;
		int threshold = (int)(k1*average+k2*deviation);
	

		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				if (src.at<uchar>(i, j) <= threshold) {
					bitImg.at<uchar>(i, j) = 255;
				}
				else {
					bitImg.at<uchar>(i, j) = 0;
				}

			}

		imshow("input image", src);
		imshow("bit image", bitImg);
		waitKey(0);
	}
}

void bitImage1(Mat src, Mat* bitImg) {
	*bitImg = src.clone();
	int height = src.rows;
	int width = src.cols;
	float average = Average(src);
	float deviation = Deviation(src, average);
	int k1 = 1, k2 = -1;
	int threshold = (int)(k1 * average + k2 * deviation);

	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++) {
			if (src.at<uchar>(i, j) <= threshold) {
				bitImg->at<uchar>(i, j) = 255;
			}
			else {
				bitImg->at<uchar>(i, j) = 0;
			}
		}
}


float Average(Mat src) {
	int sum = 0;
	int height = src.rows;
	int width = src.cols;

	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++) {
			sum += src.at<uchar>(i, j);
		}
	return (float)sum/(float)(height*width);
}

float Deviation(Mat src, float avr) {
	float sum = 0;
	int height = src.rows;
	int width = src.cols;
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++) {
			sum += (src.at<uchar>(i, j)-avr)* (src.at<uchar>(i, j) - avr);
		}

	return sqrt(sum / (float)(height * width));
}
void fill_neighbours2(Mat OriginalImage, Mat* DestinationOmage, Mat Structural, int i, int j, int trashold=100)
{

	int is = Structural.rows / 2;
	int js = Structural.cols / 2;

	int minimum = 256;

	for (int i2 = -Structural.rows / 2; i2 < Structural.rows / 2; i2++)
		for (int j2 = -Structural.cols / 2; j2 < Structural.cols / 2; j2++)
		{


			if (Structural.at<uchar>(is + i2, js + j2) <= trashold)

				if (i + i2 >= 0 && j + j2 >= 0 && i + i2 < OriginalImage.rows && j + j2 < OriginalImage.cols)
				minimum = min(minimum, OriginalImage.at<uchar>(i + i2, j + j2));

		}
	(*DestinationOmage).at<uchar>(i,j) = minimum;
}




void check_neighbours2(Mat OriginalImage, Mat* DestinationOmage, Mat Structural, int i, int j, int trashold = 100, int object = 0)
{

	int is = Structural.rows / 2;
	int js = Structural.cols / 2;

	int maxim = -1;
	if (OriginalImage.at<uchar>(i, j) > trashold)
		return;

	for (int i2 = -Structural.rows / 2; i2 < Structural.rows / 2; i2++)
		for (int j2 = -Structural.cols / 2; j2 < Structural.cols / 2; j2++)
		{


			if (i + i2 < 0 || j + j2 < 0 || i + i2 >= OriginalImage.rows || j + j2 >= OriginalImage.cols)
				maxim = 255;


			maxim = max(Structural.at<uchar>(is + i2, js + j2), maxim);



		}

	(*DestinationOmage).at<uchar>(i, j) = maxim;


}





void dilatarea(Mat imgA, Mat imgB, Mat* ImgDestination, int trashold = 100, int fundal = 255, int object = 0)
{

	(*ImgDestination) = Mat(imgA.rows, imgA.cols, CV_8UC1);

	for (int i = 0; i < imgA.rows; i++)
		for (int j = 0; j < imgA.cols; j++)
			(*ImgDestination).at<uchar>(i, j) = fundal;


	for (int i = 0; i < imgA.rows; i++)
		for (int j = 0; j < imgA.cols; j++)
			fill_neighbours2(imgA, ImgDestination, imgB, i, j, trashold);

//	imshow("before dilatation", imgA);
//	imshow("dilatated", (*ImgDestination));

}








void eroziunea(Mat imgA, Mat imgB, Mat* ImgDestination, int trashold = 100, int fundal = 255, int object = 0)
{

	(*ImgDestination) = Mat(imgA.rows, imgA.cols, CV_8UC1);
	for (int i = 0; i < imgA.rows; i++)
		for (int j = 0; j < imgA.cols; j++)
			(*ImgDestination).at<uchar>(i, j) = fundal;


	for (int i = 0; i < imgA.rows; i++)
		for (int j = 0; j < imgA.cols; j++)
			check_neighbours2(imgA, ImgDestination, imgB, i, j, trashold, object);

//	imshow("before erosion", imgA);
//	imshow("before erosion", imgB);
//	imshow("eroted", *ImgDestination);

}

void scadere(Mat ImgA, Mat ImgB, Mat* diff, int trashold = 100, int fundal = 255, int object = 0)
{

	(*diff) = Mat(ImgA.rows, ImgA.cols, CV_8UC1);


	for (int i = 0; i < ImgA.rows; i++)
		for (int j = 0; j < ImgA.cols; j++)
		{
		
				(*diff).at<uchar>(i, j) = 255+(ImgA.at<uchar>(i, j) - ImgB.at<uchar>(i, j));

		}

}


void inchidere(Mat ImgA, Mat Satructural, Mat* Inchidere, int trashhold = 100, int fundal = 255, int object = 0)
{



	Mat ImgDilatation;
	Mat ImgErosion;
	dilatarea(ImgA, Satructural, &ImgDilatation, trashhold, fundal, object);
	eroziunea(ImgDilatation, Satructural, &ImgErosion, trashhold, fundal, object);

//	imshow("before inchidere A", ImgA);
//	imshow("before dinchidere B", Satructural);
//	imshow("inchidere", ImgErosion);
	(*Inchidere) = ImgErosion;
}

void deschidere(Mat ImgA, Mat Satructural, Mat* Deschidere, int trashhold = 100, int fundal = 255, int object = 0)
{


	Mat ImgDilatation;
	Mat ImgErosion;
	eroziunea(ImgA, Satructural, &ImgErosion, trashhold, fundal, object);
	dilatarea(ImgErosion, Satructural, &ImgDilatation, trashhold, fundal, object);

	imshow("before deschidere A", ImgA);
	imshow("before dinchidere B", Satructural);
	imshow("deschidere", ImgDilatation);
	(*Deschidere) = ImgDilatation;

}
/*
void paintNeighbors(Mat* src2, int i, int j) {
	int height = src2->rows;
	int width = src2->cols;
	int aux1 = 1;

	for (int k = -aux1; k <= aux1; ++k)
		for (int r = -aux1; r <= aux1; ++r) {
			if (i + k >= 0 && i + k < height && j + r >= 0 && j + r < width) {
				src2->at<uchar>(i + k, j + r) = 0;
			}
		}
}

void dilatare1(Mat src, Mat * src2) {
	int height = src.rows;
	int width = src.cols;
	int maxValue = 220;

	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++) {
			src2->at<uchar>(i, j) = 255;
		}

	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++) {

			if (src.at<uchar>(i, j) < maxValue) {
				paintNeighbors(src2, i, j);
			}
		}
}

void dilatare() {
	char fname[MAX_PATH];
	openFileDlg(fname);
	Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
	
	Mat src2 = src.clone();
	bitImage1(src, &src2);
	
	Mat src3 = src.clone();
	dilatare1(src2, &src3);

	imshow("Initial Image", src);
	imshow("bitImage", src2);
	imshow("dilatare", src3);
	waitKey(0);
}



bool checkNeighbors(Mat* src2, int i, int j) {
	int height = src2->rows;
	int width = src2->cols;
	int maxValue = 220;
	int aux1 = 1;

	for (int k = -aux1; k <= aux1; ++k)
		for (int r = -aux1; r <= aux1; ++r) {
			if (i + k >= 0 && i + k < height && j + r >= 0 && j + r < width) {
				if (src2->at<uchar>(i + k, j + r) > maxValue)
					return false;
			}
		}
	return true;
}

void eroziunea1(Mat src, Mat* src2) {
	int height = src.rows;
	int width = src.cols;
	int maxValue = 220;


	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++) {
			src2->at<uchar>(i, j) = 255;
		}

	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++) {

			if (src.at<uchar>(i, j) < maxValue) {
				if (checkNeighbors(&src, i, j)) {
					src2->at<uchar>(i, j) = 0;
				}
			}
		}
}

void eroziunea() {
	char fname[MAX_PATH];
	openFileDlg(fname);
	Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);

	Mat src2 = src.clone();
	bitImage1(src, &src2);

	Mat src3 = src.clone();
	eroziunea1(src2, &src3);

	imshow("Initial Image", src);
	imshow("bitImage", src2);
	imshow("eroziune", src3);
	waitKey(0);
}



void deschiderea() {
	char fname[MAX_PATH];
	openFileDlg(fname);
	Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
	
	Mat src1 = src.clone();
	bitImage1(src, &src1);
	Mat src2 = src.clone();
	Mat src3 = src.clone();

	eroziunea1(src1, &src2);
	dilatare1(src2, &src3);

	imshow("Initial Image", src);
	imshow("bitImage", src1);
	imshow("deschidere", src3);
	waitKey(0);
}

void inchiderea() {
	char fname[MAX_PATH];
	openFileDlg(fname);
	Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);

	Mat src1 = src.clone();
	bitImage1(src, &src1);
	Mat src2 = src.clone();
	Mat src3 = src.clone();


	dilatare1(src1, &src2);
	eroziunea1(src2, &src3);

	imshow("Initial Image", src);
	imshow("bitImage", src1);
	imshow("inchidere", src3);
	waitKey(0);
}

void scadere(Mat src, Mat* src2, Mat* result) {
	int height = src.rows;
	int width = src.cols;
	*result = src.clone();
	int trashhold = 100;

	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++) {

			if (src.at<uchar>() < trashhold && src2->at<uchar>(i, j) < trashhold) {
				result->at<uchar>(i, j) = 255;
			}
		}
}








void extragereaConturului() {
	char fname[MAX_PATH];
	openFileDlg(fname);
	Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
	Mat src2 = src.clone();
	Mat src3 = src.clone();
	eroziunea1(src, &src2);
	scadere(src, &src2, &src3);

	imshow("First Image", src);
	imshow("eroziunea", src2);
	imshow("scaderea", src3);
	waitKey(0);
}

void extragereaConturului1(Mat src, Mat* src3) {
	//char fname[MAX_PATH];
	//openFileDlg(fname);
	//Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
	Mat src2 = src.clone();
	*src3 = src.clone();
	eroziunea1(src, &src2);
	scadere(src, &src2, src3);
}

*/
void BuildStructuralElement8(int rows, int cols, Mat* Structural, int object = 0)
{

	(*Structural) = Mat(rows, cols, CV_8UC1);

	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
			(*Structural).at<uchar>(i, j) = object;

}
void BuildStructuralElement4(int rows, int cols, Mat* Structural, int object = 0, int fundal = 255)
{

	(*Structural) = Mat(rows, cols, CV_8UC1);

	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
			(*Structural).at<uchar>(i, j) = fundal;

	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
			if (i == rows / 2 || j == cols / 2)
				(*Structural).at<uchar>(i, j) = object;

}


void testErosion()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname, IMREAD_GRAYSCALE);
		imshow("image", src);
		Mat structural;

		BuildStructuralElement8(11, 11, &structural);
		Mat iesire;
		eroziunea(src, structural, &iesire);
		waitKey();

	}


}






void tesDilatation()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname, IMREAD_GRAYSCALE);
		imshow("image", src);
		Mat structural;

		BuildStructuralElement8(11, 11, &structural);
		Mat iesire;
		dilatarea(src, structural, &iesire);
		waitKey();

	}


}


void testInchidere()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname, IMREAD_GRAYSCALE);
		imshow("image", src);
		Mat structural;

		BuildStructuralElement8(7, 7, &structural);
		Mat iesire;
		inchidere(src, structural, &iesire);
		waitKey();

	}


}
void TopHat(Mat imgA, Mat imgB, Mat* ImgDestination, int trashold = 100, int fundal = 255, int object = 0)
{
	Mat Deschidere;
	Mat Scadere;
	deschidere(imgA, imgB, &Deschidere, trashold, fundal, 0);
	scadere(imgA, Deschidere, &Scadere, trashold, fundal, 0);
	(*ImgDestination) = Scadere;

}
void BottomHat(Mat imgA, Mat imgB, Mat* ImgDestination, int trashold = 100, int fundal = 255, int object = 0)
{

	Mat Inchidere;
	Mat Scadere;
	inchidere(imgA, imgB, &Inchidere, trashold, fundal, 0);
	scadere(Inchidere, imgA, &Scadere, trashold, fundal, 0);
	(*ImgDestination) = Scadere;

}

void complement(Mat src, Mat* src2) {
	int height = src.rows;
	int width = src.cols;
	*src2 = src.clone();
	int trashhold = 100;

	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++) {
			src2->at<uchar>(i, j) = 255 - src.at<uchar>(i, j);
		}
}

void choosPointInside(Mat cont, int* i, int* j) {

}
/*
void umplereaRegiunilor() {
	char fname[MAX_PATH];
	openFileDlg(fname);
	Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);

	Mat* contur=NULL;/////
	extragereaConturului1(src, contur);

	Mat* complementContur=NULL;///
	complement(*contur, complementContur);

	Mat result = src.clone();
	int height = src.rows;
	int width = src.cols;
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++) {
			result.at<uchar>(i, j) = 255;
		}

	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++) {
			result.at<uchar>(i, j) = 255;
		}
}

*/



int main()
{
	int op;
	int ce;
	int imgRows = 3;
	int imgCols = 3;
	Mat Original;
	Mat ElementStructural;
	printf("Element Structural n4 alege 1 sau Element Structural n8 alege 2\n");
	scanf("%d", &ce);
	if(ce==1)
   BuildStructuralElement4(imgRows, imgCols, &ElementStructural);
	else
	if(ce==2)
   BuildStructuralElement8(imgRows, imgCols, &ElementStructural);

		char fname[MAX_PATH];
		while (openFileDlg(fname))
		{

			Original = imread(fname, IMREAD_GRAYSCALE);
			imshow("image", Original);
			waitKey();

			system("cls");
			printf("1 TopHat pipeline\n");
			printf("2 BottomHat pipeline\n");
			scanf("%d", &op);

			Mat Transformed;
			Mat Binarised;
			if (op == 1)
			{
				
				TopHat(Original, ElementStructural, &Transformed);
				
				//bitImage1(Transformed, &Binarised);
				imshow("original", Original);
				imshow("top hat", Transformed);
				//imshow("binarized", Binarised);
			}
			else
			if (op == 2)
			{

				
				BottomHat(Original, ElementStructural, &Transformed);
				//bitImage1(Transformed, &Binarised);
				imshow("original", Original);
				imshow("bottom hat", Transformed);
				//imshow("binarized", Binarised);
			 }
		/*	int op2;
			printf("1 Inchidere\n");
			printf("2 Deschidere\n");
			scanf("%d", &op2);
			if (op2 == 1)
			{
				Mat Inchidere;
				inchidere(Binarised, ElementStructural, &Inchidere);
				imshow("closed", Inchidere);
				waitKey();
			}
			else
			if (op2 == 2)
			{
				Mat Deschidere;
				deschidere(Binarised, ElementStructural, &Deschidere);
				imshow("opened", Deschidere);
				waitKey();


			}

         */
			waitKey();


		}

	return 0;
}