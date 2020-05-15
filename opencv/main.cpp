
#include <QtCore/QCoreApplication>
#include "stdafx.h"
#include "define.h"
#include "GetInnerRect.h"
#include <QDir>
#ifdef TEST35
typedef struct MEASURE_INFO_FLAG
{
	int       width;
	double    time;
	cv::Rect rect;
	cv::Mat resultImg;
}MEASURE_INFO;
enum IMAGE_SIZE
{
	IMAGE_WIDTH = 2048,
	IAMGE_HEIGHT = 2048
};
void measure(cv::Mat src,int threshold,int vshadow,MEASURE_INFO &measure_info);
#endif
#ifdef COURSE_02_TEST03
const char* harris_win = "Custom Harris Corners Detector";
const char* shitomasi_win = "Custom Shi-Tomasi Corners Detector";
Mat src, gray_src;
// harris corner response
Mat harris_dst, harrisRspImg;
double harris_min_rsp;
double harris_max_rsp;
// shi-tomasi corner response
Mat shiTomasiRsp;
double shitomasi_max_rsp;
double shitomasi_min_rsp;
int sm_qualitylevel = 30;
// quality level
int qualityLevel = 30;
int max_count = 100;
void CustomHarris_Demo(int, void*);
void CustomShiTomasi_Demo(int, void*);
#endif
#ifdef COURSE_02_TEST04
int max_corners = 20;
int max_count = 50;
Mat src, gray_src;
const char* output_title = "SubPixel Result";
void SubPixel_Demo(int, void*);
#endif
#ifdef COURSE_02_TEST08
Mat src, gray_src;
int current_radius = 3;
int max_count = 20;
const char* output_tt = "LBP Result";
const char* output_et = "ELBP Result";
void ELBP_Demo(int, void*);
#endif
////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[])
{
	QCoreApplication a(argc, argv);
#ifdef TEST02
	Mat image = imread("test.jpg");
	Mat invertImage;
	image.copyTo(invertImage);
	int channels = image.channels();
	int step = image.step;
	int rows = image.rows;
	int cols = image.cols*channels;
	if (image.isContinuous())
	{
		cols *= rows;
		rows = 1;
	}
	//每个像素点的每个通道255取反
	uchar* p1;
	uchar* p2;
	for (int row = 0; row < rows; row++)
	{
		p1 = image.ptr<uchar>(row);
		p2 = invertImage.ptr<uchar>(row);
		for (int col = 0; col < cols; col++)
		{
			*p2 = 255 - *p1;
			p1++;
			p2++;
		}
	}
	imshow("MyTest", image);
	imshow("My Invert Image", invertImage);
#endif
#ifdef TEST03
	Mat src, dst;
	src = imread("Test.jpg");
	if (src.empty())
	{
		return -1;
	}
	imshow("input image", src);
	double t = getTickCount();
	Mat kernel = (Mat_<char>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
	kernel /= 9;
	filter2D(src, dst, src.depth(), kernel);
	double timeconsume = (getTickCount() - t) / getTickFrequency();
	printf("time consume %.2f\n", timeconsume);
	imshow("consume image demo", dst);

	Mat testImage = imread("Test.jpg");
	imshow("test_image", testImage);
	//use Filter2D 
	Mat result;
	Mat kernel = (Mat_<char>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
	double t = getTickCount();
	filter2D(testImage, result, testImage.depth(), kernel);
	double time = (getTickCount() - t) / getTickFrequency();
	printf("time %.2f\n", time);
cv:imshow("result_image", result);
#endif
#ifdef TEST04
	Mat src = imread("Test.jpg");
	if (src.empty())
	{
		return -1;
	}
	imshow("input", src);
	Mat dst;
	cvtColor(src, dst, CV_BGR2GRAY);
	int cols = dst.cols;
	int rows = dst.rows;
	printf("rows:%d,cols:%d\n", rows, cols);
	const uchar* firstRow = dst.ptr<uchar>(0);
	printf("first pixel value:%d\n", firstRow);

	Mat M(200, 100, CV_8UC1, Scalar(127));
	Mat m1;
	m1.create(src.size(), src.type());
	m1 = Scalar(0, 0, 255);
	//	m1.data = src.data;
	imshow("output_m1", m1);

	Mat csrc;
	Mat kernel = (Mat_<char>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
	filter2D(src, csrc, src.depth(), kernel);
	imshow("output_kernel", csrc);

	Mat m2 = Mat::zeros(src.rows, src.cols, src.type());
	memcpy(m2.data, src.data, src.rows*src.step);
	imshow("output_m2", m2);
#endif
#ifdef TEST05
	Mat image = imread("TM.jpg");
	if (image.empty())
	{
		return -1;
	}
	Mat grayImg;
	cvtColor(image, grayImg, CV_BGR2GRAY);
	Mat sobelx, sobely;
	//Sobel算子求梯度
	Sobel(grayImg, sobelx, CV_32F, 1, 0);
	Sobel(grayImg, sobely, CV_32F, 0, 1, 1);
	double minVal, maxVal;
	minMaxLoc(sobely, &minVal, &maxVal);
	Mat draw;
	//	sobely.convertTo(draw, CV_8U, 255.0 / (maxVal - minVal), -minVal*255.0 / (maxVal - minVal));
	sobelx.convertTo(draw, CV_8U, 1, 255);

	//int height = image.rows;
	//int width = image.cols;
	//int channels = image.channels();
	//printf("height=%d width=%d channels=%d", height, width, channels);
	//for (int row = 0; row < height; row++)
	//{
	//	for (int col = 0; col < width; col++)
	//	{
	//		if (channels == 3)
	//		{
	//			image.at<Vec3b>(row, col)[0] = 0; // blue
	//			image.at<Vec3b>(row, col)[1] = 0; // green
	//		}
	//	}
	//}
#endif
#ifdef TEST06
	Mat src1, src2, src3;
	src1 = imread("Test.jpg");
	src2 = imread("TM.jpg");
	pyrDown(src2, src3);
	if (src1.empty() || src3.empty())
	{
		return -1;
	}
	double alpha1 = 0.5;
	double alpha2 = 0.5;
	Mat imageROI = src1(Rect(10, 20, src3.cols, src3.rows));
	addWeighted(imageROI, alpha1, src3, alpha2, 0., imageROI);
#endif
#ifdef TEST07
	Mat src, dst;
	src = imread("Test.jpg");
	if (src.empty())
	{
		return -1;
	}
	cvtColor(src, src, CV_BGR2GRAY);
	int height = src.rows;
	int width = src.cols;
	dst = Mat::zeros(src.rows, src.cols, src.type());
	float alpha = 1.2; //对比度
	float beta = 30;   //亮度
	Mat m1;
	src.convertTo(m1, CV_32F);
	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			if (src.channels() == 3)
			{
				float b = m1.at<Vec3f>(row, col)[0];// blue
				float g = m1.at<Vec3f>(row, col)[1]; // green
				float r = m1.at<Vec3f>(row, col)[2]; // red

				// output
				dst.at<Vec3b>(row, col)[0] = saturate_cast<uchar>(b*alpha + beta);
				dst.at<Vec3b>(row, col)[1] = saturate_cast<uchar>(g*alpha + beta);
				dst.at<Vec3b>(row, col)[2] = saturate_cast<uchar>(r*alpha + beta);
			}
			else if (src.channels() == 1) {
				float v = src.at<uchar>(row, col);
				dst.at<uchar>(row, col) = saturate_cast<uchar>(v*alpha + beta);
			}
		}
	}
	imshow("output", dst);
#endif
#ifdef TEST08
	Mat bgImage = imread("Test.jpg");
	if (bgImage.empty())
	{
		return -1;
	}
	//line
	Point p1 = Point(20, 30);
	Point p2;
	p2.x = 400;
	p2.y = 400;
	Scalar color1 = Scalar(0, 0, 255);
	line(bgImage, p1, p2, color1, 1, LINE_AA);
	//rectangle
	Rect rect(200, 100, 300, 300);
	Scalar color2 = Scalar(255, 0, 0);
	rectangle(bgImage, rect, color2, 2, LINE_8);
	//ellipse
	Scalar color3 = Scalar(0, 255, 0);
	Point pt = Point(bgImage.cols / 2, bgImage.rows / 2);
	Size size = Size(bgImage.cols / 4, bgImage.rows / 8);
	ellipse(bgImage, pt, size, 90, 0, 360, color3, 2, LINE_8);
	//circle
	Scalar color4 = Scalar(0, 255, 255);
	Point center = Point(bgImage.cols / 2, bgImage.rows / 2);
	circle(bgImage, center, 150, color4, 2, 8);
#endif
#ifdef TEST09
	Mat src;
	src = imread("Test.jpg");
	if (src.empty())
	{
		return -1;
	}
	Mat bImg;
	blur(src, bImg, Size(11, 11), Point(-1, -1));
	Mat gsImg;
	GaussianBlur(src, gsImg, Size(11, 11), 3, 3);
#endif
#ifdef TEST10
	Mat src;
	src = imread("Test.jpg");
	if (src.empty())
	{
		return -1;
	}
	Mat bfilter;
	bilateralFilter(src, bfilter, 15, 100, 5);
	Mat kfilter;
	Mat kernel = (Mat_<int>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
	filter2D(src, kfilter, -1, kernel, Point(-1, -1), 0);
#endif
#ifdef TEST11
	Mat Ddst, Edst;
	Mat src = imread("TM.jpg");
	if (src.empty())
	{
		return -1;
	}
	Mat structureElement = getStructuringElement(MORPH_RECT, Size(5, 5), Point(-1, -1));
	dilate(src, Ddst, structureElement, Point(-1, -1), 1);//膨胀
	erode(src, Edst, structureElement);//腐蚀
#endif
#ifdef TEST12

	Mat src = imread("TM.jpg");
	if (src.empty())
	{
		return -1;
	}
	Mat pGray;
	cvtColor(src, pGray, CV_BGR2GRAY);
	Mat img_threshold;
	threshold(pGray, img_threshold, 0, 255, CV_THRESH_OTSU);
	//获取结构元素(内核矩阵)，包括结构元素的大小及形状
	Mat element = getStructuringElement(MORPH_RECT, Size(5, 5), Point(-1, -1));
	Mat openImg, closeImg, gradientImg, tophatImg, blackhatImg;
	/*
	高级形态学变换：
	开运算：
	先腐蚀，再膨胀，可清除一些小东西(亮的)，放大局部低亮度的区域
	闭运算：
	先膨胀，再腐蚀，可清除小黑点
	形态学梯度：
	膨胀图与腐蚀图之差，提取物体边缘
	顶帽：
	原图像与开运算图之差，突出原图像中比周围亮的区域
	黑帽：
	闭运算图与原图像之差，突出原图像中比周围暗的区域
	*/
	/*
	morphologyEx函数利用基本的膨胀和腐蚀技术，来执行更加高级形态学变换
	src
	输入图像，图像位深应该为以下五种之一：CV_8U, CV_16U,CV_16S, CV_32F 或CV_64F。
	dst
	输出图像，需和源图片保持一样的尺寸和类型。
	op
	表示形态学运算的类型：
	MORPH_OPEN – 开运算（Opening operation）
	MORPH_CLOSE – 闭运算（Closing operation）
	MORPH_GRADIENT - 形态学梯度（Morphological gradient）
	MORPH_TOPHAT - 顶帽（Top hat）
	MORPH_BLACKHAT - 黑帽（Black hat）
	kernel
	形态学运算的内核。为NULL，使用参考点位于中心3x3的核。一般使用函数getStructuringElement配合这个参数的使用，
	kernel参数填保存getStructuringElement返回值的Mat类型变量。
	anchor
	锚的位置，其有默认值（-1，-1），表示锚位于中心。
	iterations
	迭代使用函数的次数，默认值为1。
	borderType
	用于推断图像外部像素的某种边界模式。注意它有默认值BORDER_CONSTANT。
	borderValue
	当边界为常数时的边界值，有默认值morphologyDefaultBorderValue()，
	一般我们不用去管他。需要用到它时，可以看官方文档中的createMorphologyFilter()函数得到更详细的解释。
	*/
	morphologyEx(img_threshold, openImg, CV_MOP_OPEN, element); //先腐蚀，再膨胀，可清除一些小亮点，放大局部低亮度的区域
	morphologyEx(img_threshold, closeImg, CV_MOP_CLOSE, element); //先膨胀，再腐蚀，可清除小黑点
	morphologyEx(img_threshold, gradientImg, CV_MOP_GRADIENT, element); //膨胀图与腐蚀图之差，提取物体边缘
	morphologyEx(img_threshold, tophatImg, CV_MOP_TOPHAT, element);//原图像-开运算图，突出原图像中比周围亮的区域
	morphologyEx(img_threshold, blackhatImg, CV_MOP_BLACKHAT, element);//闭运算图-原图像，突出原图像中比周围暗的区域
#endif
#ifdef TEST13
	Mat src = imread("TM.jpg");
	if (src.empty())
	{
		return -1;
	}
	Mat pGray;
	cvtColor(src, pGray, CV_BGR2GRAY);
	Mat binImg;
	adaptiveThreshold(~pGray, binImg, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, -2);
	// 水平结构元素
	Mat hline = getStructuringElement(MORPH_RECT, Size(src.cols / 16, 1), Point(-1, -1));
	// 垂直结构元素
	Mat vline = getStructuringElement(MORPH_RECT, Size(1, src.rows / 16), Point(-1, -1));
	// 矩形结构
	Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
	Mat temp;
	Mat binImg2;
	//erode(binImg, temp, kernel);
	//dilate(temp, binImg2, kernel);
	morphologyEx(binImg, binImg2, CV_MOP_OPEN, vline);

	Mat dstImg;
	bitwise_not(binImg2, binImg2);
	imshow("Final Result", binImg2);
#endif
#ifdef TEST14

	Mat src = imread("TM.jpg");
	if (src.empty())
	{
		return -1;
	}
	Mat s_up, s_down;
	// 上采样
	pyrUp(src, s_up, Size(src.cols * 2, src.rows * 2));
	// 降采样
	pyrDown(src, s_down, Size(src.cols / 2, src.rows / 2));
	// DOG
	Mat gray_src, g1, g2, dogImg;
	cvtColor(src, gray_src, CV_BGR2GRAY);
	GaussianBlur(gray_src, g1, Size(5, 5), 0, 0);
	GaussianBlur(g1, g2, Size(5, 5), 0, 0);
	subtract(g1, g2, dogImg, Mat());

	// 归一化显示
	normalize(dogImg, dogImg, 255, 0, NORM_MINMAX);
	imshow("DOG Image", dogImg);
#endif
#ifdef TEST15
	Mat src = imread("Test.jpg");
	if (src.empty())
	{
		return -1;
	}
	Mat gray_src, dst, binImg;
	cvtColor(src, gray_src, CV_BGR2GRAY);
	threshold(gray_src, dst, 100, 255, THRESH_BINARY);
	adaptiveThreshold(~gray_src, binImg, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, -2);
#endif
#ifdef TEST16
	Mat src = imread("TM.jpg");
	if (src.empty())
	{
		return -1;
	}
	//	 Sobel X 方向
	Mat sobel_x;
	Mat kernel_x = (Mat_<int>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
	filter2D(src, sobel_x, -1, kernel_x, Point(-1, -1), 0.0);

	//	 Sobel Y 方向
	Mat sobel_y;
	Mat kernel_y = (Mat_<int>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
	filter2D(src, sobel_y, -1, kernel_y, Point(-1, -1), 0.0);

	//	 拉普拉斯算子
	Mat lap;
	Mat kernel_l = (Mat_<int>(3, 3) << 0, -1, 0, -1, 4, -1, 0, -1, 0);
	filter2D(src, lap, -1, kernel_l, Point(-1, -1), 0.0);

	//自定义滤波核
	int c = 0;
	int index = 0;
	int ksize = 0;
	Mat dst;
	while (true)
	{
		c = waitKey(500);
		if ((char)c == 27) {// ESC 
			break;
		}
		ksize = 5 + (index % 8) * 2;
		Mat kernel = Mat::ones(Size(ksize, ksize), CV_32F) / (float)(ksize * ksize);
		filter2D(src, dst, -1, kernel, Point(-1, -1));
		index++;
		imshow("outImg", dst);
	}
#endif
#ifdef TEST17
	Mat dst;
	Mat src = imread("Test.jpg");
	if (src.empty())
	{
		return -1;
	}
	int top = (int)(0.05*src.rows);
	int bottom = (int)(0.05*src.rows);
	int left = (int)(0.05*src.cols);
	int right = (int)(0.05*src.cols);
	RNG rng(12345);
	int borderType = BORDER_DEFAULT;

	int c = 0;
	while (true)
	{
		c = waitKey(500);
		// ESC
		if ((char)c == 27)
		{
			break;
		}
		if ((char)c == 'r')
		{
			borderType = BORDER_REPLICATE;//复制法，也就是复制最边缘像素。
		}
		else if ((char)c == 'w')
		{
			borderType = BORDER_WRAP;
		}
		else if ((char)c == 'c')
		{
			borderType = BORDER_CONSTANT;//常量法。
		}
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		//copyMakeBorder填充图像边界
		copyMakeBorder(src, dst, top, bottom, left, right, borderType, color);
		imshow("out", dst);
	}
#endif
#ifdef TEST18
	Mat src = imread("TM.jpg");
	if (src.empty())
	{
		return -1;
	}
	Mat gray_src;
	GaussianBlur(src, src, Size(3, 3), 0, 0);
	cvtColor(src, gray_src, CV_BGR2GRAY);
	Mat xgrad, ygrad;
	Scharr(gray_src, xgrad, CV_16S, 1, 0);
	Scharr(gray_src, ygrad, CV_16S, 0, 1);

	// Sobel(gray_src, xgrad, CV_16S, 1, 0, 3);
	// Sobel(gray_src, ygrad, CV_16S, 0, 1, 3);
	//convertScaleAbs操作可实现图像增强等相关操作的快速运算
	convertScaleAbs(xgrad, xgrad);
	convertScaleAbs(ygrad, ygrad);
	Mat xygrad = Mat(xgrad.size(), xgrad.type());
	printf("type : %d\n", xgrad.type());
	int width = xgrad.cols;
	int height = ygrad.rows;
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			int xg = xgrad.at<uchar>(row, col);
			int yg = ygrad.at<uchar>(row, col);
			int xy = xg + yg;
			xygrad.at<uchar>(row, col) = saturate_cast<uchar>(xy);
		}
	}
	//	addWeighted(xgrad, 0.5, ygrad, 0.5, 0, xygrad);
	imshow("OUTPUT_TITLE", xygrad);
#endif
#ifdef TEST19
	Mat src = imread("TM.jpg");
	if (src.empty())
	{
		return -1;
	}
	Mat gray_src;
	GaussianBlur(src, src, Size(3, 3), 0, 0);
	cvtColor(src, gray_src, CV_BGR2GRAY);
	Mat lap_img;
	Laplacian(gray_src, lap_img, CV_16S, 3);
	Mat edge_image;
	convertScaleAbs(lap_img, edge_image);
	Mat image;
	threshold(edge_image, image, 0, 255, THRESH_OTSU | THRESH_BINARY);
#endif
#ifdef TEST20
	Mat src = imread("TM.jpg");
	if (src.empty())
	{
		return -1;
	}
	Mat gray_src;
	cvtColor(src, gray_src, CV_BGR2GRAY);
	blur(gray_src, gray_src, Size(3, 3), Point(-1, -1), BORDER_DEFAULT);
	Mat edge_output;
	int t1_value = 50;
	Canny(gray_src, edge_output, t1_value, t1_value * 2, 3, false);
#endif
#ifdef TEST21
	Mat src = imread("TM.jpg");
	if (src.empty())
	{
		return -1;
	}
	Mat src_gray, dst;
	Canny(src, src_gray, 150, 200);
	cvtColor(src_gray, dst, CV_GRAY2BGR);
	vector<Vec2f> lines;
	HoughLines(src_gray, lines, 1, CV_PI / 180, 150, 0, 0);
	for (size_t i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0]; // 极坐标中的r长度
		float theta = lines[i][1]; // 极坐标中的角度
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;
		// 转换为平面坐标的四个点
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		line(dst, pt1, pt2, Scalar(0, 0, 255), 1, CV_AA);
	}
	Mat dst2 = dst.clone();
	vector<Vec4f> plines;
	HoughLinesP(src_gray, plines, 1, CV_PI / 180.0, 10, 0, 10);
	Scalar color = Scalar(0, 0, 255);
	for (size_t i = 0; i < plines.size(); i++)
	{
		Vec4f hline = plines[i];
		line(dst2, Point(hline[0], hline[1]), Point(hline[2], hline[3]), color, 1, LINE_AA);
	}
#endif
#ifdef TEST22
	Mat src = imread("Test.jpg");
	if (src.empty())
	{
		return -1;
	}
	Mat moutput;
	medianBlur(src, moutput, 3);
	cvtColor(moutput, moutput, CV_BGR2GRAY);
	// 霍夫圆检测
	Mat dst;
	vector<Vec3f> pcircles;
	/*
	InputArray： 输入图像，数据类型一般用Mat型即可，需要是8位单通道灰度图像
	OutputArray：存储检测到的圆的输出矢量
	method：使用的检测方法，目前opencv只有霍夫梯度法一种方法可用，该参数填HOUGH_GRADIENT即可（opencv 4.1.0下）
	dp：double类型的dp，用来检测圆心的累加器图像的分辨率于输入图像之比的倒数，且此参数允许创建一个比输入图像分辨率低的累加器。上述文字不好理解的话，来看例子吧。例如，如果dp= 1时，累加器和输入图像具有相同的分辨率。如果dp=2，累加器便有输入图像一半那么大的宽度和高度。
	minDist：为霍夫变换检测到的圆的圆心之间的最小距离
	param1：它是第三个参数method设置的检测方法的对应的参数。对当前唯一的方法霍夫梯度法CV_HOUGH_GRADIENT，它表示传递给canny边缘检测算子的高阈值，而低阈值为高阈值的一半。
	param2：也是第三个参数method设置的检测方法的对应的参数，对当前唯一的方法霍夫梯度法HOUGH_GRADIENT，它表示在检测阶段圆心的累加器阈值。它越小的话，就可以检测到更多根本不存在的圆，而它越大的话，能通过检测的圆就更加接近完美的圆形了。
	minRadius：表示圆半径的最小值
	maxRadius：表示圆半径的最大值
	*/
	HoughCircles(moutput, pcircles, CV_HOUGH_GRADIENT, 1, 10, 100, 40, 5, 50);
	src.copyTo(dst);
	for (size_t i = 0; i < pcircles.size(); i++) {
		Vec3f cc = pcircles[i];
		circle(dst, Point(cc[0], cc[1]), cc[2], Scalar(0, 0, 255), 1, LINE_AA);
		circle(dst, Point(cc[0], cc[1]), 2, Scalar(198, 23, 155), 1, LINE_AA);
	}
#endif
#ifdef TEST23
	Mat src, dst, map_x, map_y;
	src = imread("TH.jpg");
	if (src.empty())
	{
		return -1;
	}
	map_x.create(src.size(), CV_32FC1);
	map_y.create(src.size(), CV_32FC1);
	int index = 0;
	int c = 0;
	while (true)
	{
		c = waitKey(1000);
		if ((char)c == 27)
		{
			break;
		}
		index = c % 4;
		/*map_x,map_y*/
		for (int row = 0; row < src.rows; row++)
		{
			for (int col = 0; col < src.cols; col++)
			{
				switch (index)
				{
					//index = 0 ，图像的行跟列为为原来的1/2。
					//index = 1，为左右翻转（列变换，行不变）
					//index = 2，为上下翻转（行变换，列不变）
					//index = 3，为中心旋转
				case 0:
					if (col > (src.cols * 0.25) && col <= (src.cols*0.75) && row > (src.rows*0.25) && row <= (src.rows*0.75))
					{
						map_x.at<float>(row, col) = 2 * (col - (src.cols*0.25));
						map_y.at<float>(row, col) = 2 * (row - (src.rows*0.25));
					}
					else
					{
						map_x.at<float>(row, col) = 0;
						map_y.at<float>(row, col) = 0;
					}
					break;
				case 1:
					map_x.at<float>(row, col) = (src.cols - col - 1);
					map_y.at<float>(row, col) = row;
					break;
				case 2:
					map_x.at<float>(row, col) = col;
					map_y.at<float>(row, col) = (src.rows - row - 1);
					break;
				case 3:
					map_x.at<float>(row, col) = (src.cols - col - 1);
					map_y.at<float>(row, col) = (src.rows - row - 1);
					break;
				}
			}
		}
		/*********************/
		remap(src, dst, map_x, map_y, INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 255, 255));
		imshow("OUTPUT_TITLE", dst);
	}
#endif
#ifdef TEST24
	Mat src, dst;
	src = imread("TH.jpg");
	if (src.empty())
	{
		return -1;
	}
	cvtColor(src, src, CV_BGR2GRAY);
	equalizeHist(src, dst);
	imshow("outImg", dst);
#endif
#ifdef TEST25
	Mat src, dst;
	src = imread("Test.jpg");
	if (src.empty())
	{
		return -1;
	}
	//分通道显示
	vector<Mat> bgr_planes;
	split(src, bgr_planes);
	imshow("B channel", bgr_planes[0]);
	imshow("G channel", bgr_planes[1]);
	imshow("R channel", bgr_planes[2]);
	// 计算直方图
	int histSize = 256;
	float range[] = { 0, 256 };
	const float *histRanges = { range };
	Mat b_hist, g_hist, r_hist;
	calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRanges, true, false);
	calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRanges, true, false);
	calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRanges, true, false);
	// 归一化
	int hist_h = 400;
	int hist_w = 512;
	int bin_w = hist_w / histSize;
	Mat histImage(hist_w, hist_h, CV_8UC3, Scalar(0, 0, 0));
	normalize(b_hist, b_hist, 0, hist_h, NORM_MINMAX, -1, Mat());
	normalize(g_hist, g_hist, 0, hist_h, NORM_MINMAX, -1, Mat());
	normalize(r_hist, r_hist, 0, hist_h, NORM_MINMAX, -1, Mat());
	// render histogram chart
	for (int i = 1; i < histSize; i++) {
		line(histImage, Point((i - 1)*bin_w, hist_h - cvRound(b_hist.at<float>(i - 1))),
			Point((i)*bin_w, hist_h - cvRound(b_hist.at<float>(i))), Scalar(255, 0, 0), 2, LINE_AA);

		line(histImage, Point((i - 1)*bin_w, hist_h - cvRound(g_hist.at<float>(i - 1))),
			Point((i)*bin_w, hist_h - cvRound(g_hist.at<float>(i))), Scalar(0, 255, 0), 2, LINE_AA);

		line(histImage, Point((i - 1)*bin_w, hist_h - cvRound(r_hist.at<float>(i - 1))),
			Point((i)*bin_w, hist_h - cvRound(r_hist.at<float>(i))), Scalar(0, 0, 255), 2, LINE_AA);
	}
	imshow("OUTPUT_T", histImage);
#endif
#ifdef TEST26
	/*
	（1）图像相似度比较
        如果我们有两张图像，并且这两张图像的直方图一样，或者有极高的相似度，那么在一定程度上，我们可以认为这两幅图是一样的，这就是直方图比较的应用之一。
（2）分析图像之间关系
        两张图像的直方图反映了该图像像素的分布情况，可以利用图像的直方图，来分析两张图像的关系。
*/
	Mat base, test1;
	Mat hsvbase, hsvtest1;
	base = imread("Test.jpg");
	if (!base.data) {
		printf("could not load image...\n");
		return -1;
	}
	test1 = imread("Test1.jpg");
	cvtColor(base, hsvbase, CV_BGR2HSV);
	cvtColor(test1, hsvtest1, CV_BGR2HSV);
	int h_bins = 50; int s_bins = 60;
	int histSize[] = { h_bins, s_bins };
	// hue varies from 0 to 179, saturation from 0 to 255     
	float h_ranges[] = { 0, 180 };
	float s_ranges[] = { 0, 256 };
	const float* ranges[] = { h_ranges, s_ranges };
	// Use the o-th and 1-st channels     
	int channels[] = { 0, 1 };
	MatND hist_base;
	MatND hist_test1;
	calcHist(&hsvbase, 1, channels, Mat(), hist_base, 2, histSize, ranges, true, false);
	normalize(hist_base, hist_base, 0, 1, NORM_MINMAX, -1, Mat());

	calcHist(&hsvtest1, 1, channels, Mat(), hist_test1, 2, histSize, ranges, true, false);
	normalize(hist_test1, hist_test1, 0, 1, NORM_MINMAX, -1, Mat());

	double basebase = compareHist(hist_base, hist_base, CV_COMP_CORREL);
	double basetest1 = compareHist(hist_base, hist_test1, CV_COMP_CORREL);

	putText(base, QString::number(basebase).toStdString(), Point(50, 50), CV_FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255), 2, LINE_AA);
	putText(test1, QString::number(basetest1).toStdString(), Point(50, 50), CV_FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255), 2, LINE_AA);

	imshow("base", base);
	imshow("test1", test1);
#endif
#ifdef TEST27
	/*	std::vector<cv::Point2f> defectLocation;
		Point2f p1(1, 20);
		defectLocation.push_back(p1);
		Point p2 (20, 20);
		defectLocation.push_back(p2);
		Point p3 (20, 40);
		defectLocation.push_back(p3);
		Point p4(1, 40);
		defectLocation.push_back(p4);

		Mat Pt = (Mat_<char>(2,2) << defectLocation[0].x + defectLocation[0].y, defectLocation[1].x + defectLocation[1].y, defectLocation[2].x + defectLocation[2].y, defectLocation[3].x + defectLocation[3].y);
		double minVal, maxVal;
		cv::Point minLocation, maxLocation;
		cv::minMaxLoc(Pt, &minVal, &maxVal, &minLocation, &maxLocation, cv::Mat());
		int min = minLocation.x * 2 + minLocation.y;
		float itempx = pow(abs(defectLocation[min].x - defectLocation[(min + 1) % 4].x), 2);
		float itempy = pow(abs(defectLocation[min].y - defectLocation[(min + 1) % 4].y), 2);
		int iwidth = sqrt(itempx + itempy)*0.12;*/
#endif
#ifdef TEST28
		// 待检测图像
	Mat src = imread("TH.jpg");
	// 模板图像
	Mat temp = imread("TH_Test.jpg");
	if (src.empty() || temp.empty()) {
		printf("could not load image...\n");
		return -1;
	}
	int match_method = TM_CCOEFF_NORMED;
	Mat result;
	matchTemplate(src, temp, result, match_method, Mat());
	//	normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());
	Point minLoc, maxLoc, temLoc;
	double min, max;
	minMaxLoc(result, &min, &max, &minLoc, &maxLoc, Mat());
	if (match_method == TM_SQDIFF || match_method == TM_SQDIFF_NORMED) {
		temLoc = minLoc;
	}
	else {
		temLoc = maxLoc;
	}
	// 绘制矩形
	Mat dst = src.clone();
	rectangle(dst, Rect(temLoc.x, temLoc.y, temp.cols, temp.rows), Scalar(0, 0, 255), 2, 8);
	rectangle(result, Rect(temLoc.x, temLoc.y, temp.cols, temp.rows), Scalar(0, 0, 255), 2, 8);
#endif
#ifdef TEST29
	Mat src = imread("TH.jpg");
	if (!src.data)
	{
		cout << "could not load image !";
		return -1;
	}
	Mat src_gray;
	cvtColor(src, src_gray, CV_BGR2GRAY);
	blur(src_gray, src_gray, Size(23, 23));
	Mat bin_output;
	threshold(src_gray, bin_output, 10, 255, THRESH_BINARY);
	vector<vector<Point>> contours;
	vector<Vec4i> hierachy;
	findContours(bin_output, contours, hierachy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
	Mat dst = Mat::zeros(src.size(), CV_8UC3);
	RNG rng(12345);
	for (size_t i = 0; i < contours.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(bin_output, contours, i, color, 2, 8, hierachy, 0, Point(0, 0));
	}
#endif
#ifdef TEST30
	Mat src = imread("TH.jpg");
	if (!src.data)
	{
		cout << "could not load image !";
		waitKey(0);
		return -1;
	}

	Mat src_gray;
	cvtColor(src, src_gray, CV_BGR2GRAY);
	blur(src_gray, src_gray, Size(23, 23));
	Mat src_copy = src.clone();
	Mat threshold_output;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	threshold(src_gray, threshold_output, 50, 255, THRESH_BINARY_INV);
	findContours(threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	vector<vector<Point> >hull(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		convexHull(Mat(contours[i]), hull[i], false);
	}
	RNG rng(12345);
	for (int i = 0; i < contours.size(); i++)
	{
		Scalar color1 = Scalar(255, 0, 0);
		Scalar color2 = Scalar(0, 255, 0);
		drawContours(src_copy, contours, i, color1, 2, 8, vector<Vec4i>(), 0, Point());
		drawContours(src_copy, hull, i, color2, 2, 8, vector<Vec4i>(), 0, Point());
	}
#endif
#ifdef TEST31
	Mat src = imread("123.bmp");
	if (!src.data)
	{
		cout << "could not load image !";
		waitKey(0);
		return -1;
	}
	Mat src_gray;
	cv::cvtColor(src, src_gray, CV_BGR2GRAY);
	cv::blur(src_gray, src_gray, Size(23, 23));
	Mat threshold_output;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	cv::threshold(src_gray, threshold_output, 50, 255, THRESH_BINARY_INV);
	cv::findContours(threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	vector<vector<Point>> contours_ploy(contours.size());
	vector<Rect> ploy_rects(contours.size());
	vector<Point2f> ccs(contours.size());
	vector<float> radius(contours.size());

	vector<RotatedRect> minRects(contours.size());
	vector<RotatedRect> myellipse(contours.size());

	for (size_t i = 0; i < contours.size(); i++)
	{
		approxPolyDP(Mat(contours[i]), contours_ploy[i], 3, true);
		ploy_rects[i] = boundingRect(contours_ploy[i]);
		minEnclosingCircle(contours_ploy[i], ccs[i], radius[i]);
		if (contours_ploy[i].size() > 5)
		{
			myellipse[i] = fitEllipse(contours_ploy[i]);
			minRects[i] = minAreaRect(contours_ploy[i]);
		}
	}

	// draw it
	Mat drawImg = Mat::zeros(src.size(), src.type());
	Point2f pts[4];
	RNG rng(12345);
	for (size_t t = 0; t < contours.size(); t++) {
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		//rectangle(drawImg, ploy_rects[t], color, 2, 8);
		//circle(drawImg, ccs[t], radius[t], color, 2, 8);
		if (contours_ploy[t].size() > 5)
		{
			ellipse(threshold_output, myellipse[t], color, 1, 8);
			minRects[t].points(pts);
			for (int r = 0; r < 4; r++)
			{
				line(threshold_output, pts[r], pts[(r + 1) % 4], color, 1, 8);
			}
		}
	}
#endif
#ifdef TEST32
	Mat src = imread("./test/2.bmp");
	if (!src.data)
	{
		cout << "could not load image !";
		waitKey(0);
		return -1;
	}
	double t=cv::getTickCount();
	Mat src_gray;
	cv::cvtColor(src, src_gray, CV_BGR2GRAY);
	cv::blur(src_gray, src_gray, Size(11, 11));
	Mat threshold_output;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	Mat outPut;
	cv::threshold(src_gray, outPut, 10, 255, THRESH_BINARY);
	Mat structureElement = getStructuringElement(MORPH_RECT, Size(21,21), Point(-1, -1));
	cv::dilate(outPut, threshold_output, structureElement, Point(-1, -1), 1);
	cv::erode(threshold_output, threshold_output, structureElement);

	
	findContours(threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	vector<Moments> contours_moments(contours.size());
	vector<Point2f> ccs(contours.size());
	for (size_t i = 0; i < contours.size(); i++)
	{
		//moments 计算图像矩
		contours_moments[i] = moments(contours[i]);
		//中心点/能量质心
		ccs[i] = Point(static_cast<float>(contours_moments[i].m10 / contours_moments[i].m00), static_cast<float>(contours_moments[i].m01 / contours_moments[i].m00));
	}
	Mat drawImg;
	src.copyTo(drawImg);
	RNG rng(1234);
	for (size_t i = 0; i < contours.size(); i++)
	{
		//if (contours[i].size() < 100)
		//{
		//	continue;
		//}
		Scalar color1 = Scalar(255, 0, 0);
		Scalar color2 = Scalar(0,0, 255);
		printf("center point x : %.2f y : %.2f\n", ccs[i].x, ccs[i].y);
		//contourArea 轮廓面积，arcLength 轮廓周长
		printf("contours %d area : %.2f   arc length : %.2f\n", i, contourArea(contours[i]), arcLength(contours[i], true));
		drawContours(drawImg, contours, i, color1, 2, 8, hierarchy, 0, Point(0, 0));
		circle(drawImg, ccs[i], 5, color2, 5, 8);
	}
	waitKey(0);
#endif
#ifdef TEST33
	const int r = 100;
	Mat src = Mat::zeros(r * 4, r * 4, CV_8UC1);

	vector<Point2f> vert(6);
	vert[0] = Point(3 * r / 2, static_cast<int>(1.34*r));
	vert[1] = Point(1 * r, 2 * r);
	vert[2] = Point(3 * r / 2, static_cast<int>(2.866*r));
	vert[3] = Point(5 * r / 2, static_cast<int>(2.866*r));
	vert[4] = Point(3 * r, 2 * r);
	vert[5] = Point(5 * r / 2, static_cast<int>(1.34*r));

	for (int i = 0; i < 6; i++)
	{
		line(src, vert[i], vert[(i + 1) % 6], Scalar(255), 3, 8, 0);
	}

	vector<vector<Point>> contours;
	vector<Vec4i> hierachy;
	Mat csrc;
	src.copyTo(csrc);
	findContours(csrc, contours, hierachy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
	Mat raw_dist = Mat::zeros(csrc.size(), CV_32FC1);
	for (int row = 0; row < raw_dist.rows; row++)
	{
		for (int col = 0; col < raw_dist.cols; col++)
		{
			double dist = pointPolygonTest(contours[0], Point2f(static_cast<float>(col), static_cast<float>(row)), true);
			raw_dist.at<float>(row, col) = static_cast<float>(dist);
		}
	}

	double minValue, maxValue;
	minMaxLoc(raw_dist, &minValue, &maxValue, 0, 0, Mat());
	Mat drawImg = Mat::zeros(src.size(), CV_8UC3);
	for (int row = 0; row < drawImg.rows; row++)
	{
		for (int col = 0; col < drawImg.cols; col++)
		{
			float dist = raw_dist.at<float>(row, col);
			if (dist > 0) {
				drawImg.at<Vec3b>(row, col)[0] = (uchar)(abs(1.0 - (dist / maxValue)) * 255);
			}
			else if (dist < 0) {
				drawImg.at<Vec3b>(row, col)[2] = (uchar)(abs(1.0 - (dist / minValue)) * 255);
			}
			else {
				drawImg.at<Vec3b>(row, col)[0] = (uchar)(abs(255 - dist));
				drawImg.at<Vec3b>(row, col)[1] = (uchar)(abs(255 - dist));
				drawImg.at<Vec3b>(row, col)[2] = (uchar)(abs(255 - dist));
			}
		}
	}

	const char* output_win = "point polygon test demo";
	char input_win[] = "input image";
	namedWindow(input_win, CV_WINDOW_AUTOSIZE);
	namedWindow(output_win, CV_WINDOW_AUTOSIZE);

	imshow(input_win, src);
	imshow(output_win, drawImg);
#endif
#ifdef TEST34
	char input_win[] = "input image";
	char watershed_win[] = "watershed segmentation demo";
	Mat src = imread("TH.jpg");
	// Mat src = imread("D:/kuaidi.jpg");
	if (src.empty()) {
		printf("could not load image...\n");
		return -1;
	}
	namedWindow(input_win, CV_WINDOW_AUTOSIZE);
	imshow(input_win, src);
	// 1. change background
	for (int row = 0; row < src.rows; row++)
	{
		for (int col = 0; col < src.cols; col++)
		{
			if (src.at<Vec3b>(row, col) == Vec3b(255, 255, 255))
			{
				src.at<Vec3b>(row, col)[0] = 0;
				src.at<Vec3b>(row, col)[1] = 0;
				src.at<Vec3b>(row, col)[2] = 0;
			}
		}
	}
	namedWindow("black background", CV_WINDOW_AUTOSIZE);
	imshow("black background", src);

	// sharpen图像锐化
	Mat kernel = (Mat_<float>(3, 3) << 1, 1, 1, 1, -8, 1, 1, 1, 1);
	Mat imgLaplance;
	Mat sharpenImg = src;
	filter2D(src, imgLaplance, CV_32F, kernel, Point(-1, -1), 0, BORDER_DEFAULT);
	src.convertTo(sharpenImg, CV_32F);
	Mat sharpImg = sharpenImg - imgLaplance;

	sharpImg.convertTo(sharpImg, CV_8UC3);
	imgLaplance.convertTo(imgLaplance, CV_8UC3);
	imshow("sharpen图像锐化 image", sharpImg);
	// src = resultImg; // copy back

	// convert to binary
	Mat binaryImg, resultImg;
	cvtColor(src, resultImg, CV_BGR2GRAY);
	threshold(resultImg, binaryImg, 40, 255, THRESH_BINARY | THRESH_OTSU);
	imshow("binary image", binaryImg);

	Mat distImg;
	/*
	distanceTransform方法用于计算图像中每一个非零点距离离自己最近的零点的距离，
	distanceTransform的第二个Mat矩阵参数dst保存了每一个点与最近的零点的距离信息，
	图像上越亮的点，代表了离零点的距离越远。
	*/
	distanceTransform(binaryImg, distImg, DIST_L1, 3, 5);
	normalize(distImg, distImg, 0, 1, NORM_MINMAX);
	imshow("distance result", distImg);

	// binary again
	threshold(distImg, distImg, .4, 1, THRESH_BINARY);
	Mat k1 = Mat::ones(13, 13, CV_8UC1);
	erode(distImg, distImg, k1, Point(-1, -1));
	imshow("distance binary image", distImg);

	// markers 
	Mat dist_8u;
	distImg.convertTo(dist_8u, CV_8U);
	vector<vector<Point>> contours;
	findContours(dist_8u, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));

	// create makers
	Mat markers = Mat::zeros(src.size(), CV_32SC1);
	for (size_t i = 0; i < contours.size(); i++) {
		drawContours(markers, contours, static_cast<int>(i), Scalar::all(static_cast<int>(i) + 1), -1);
	}
	circle(markers, Point(5, 5), 3, Scalar(255, 255, 255), -1);
	imshow("my markers", markers * 1000);

	// perform watershed
	watershed(src, markers);
	Mat mark = Mat::zeros(markers.size(), CV_8UC1);
	markers.convertTo(mark, CV_8UC1);
	bitwise_not(mark, mark, Mat());
	imshow("watershed image", mark);

	// generate random color
	vector<Vec3b> colors;
	for (size_t i = 0; i < contours.size(); i++) {
		int r = theRNG().uniform(0, 255);
		int g = theRNG().uniform(0, 255);
		int b = theRNG().uniform(0, 255);
		colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
	}

	// fill with color and display final result
	Mat dst = Mat::zeros(markers.size(), CV_8UC3);
	for (int row = 0; row < markers.rows; row++) {
		for (int col = 0; col < markers.cols; col++) {
			int index = markers.at<int>(row, col);
			if (index > 0 && index <= static_cast<int>(contours.size())) {
				dst.at<Vec3b>(row, col) = colors[index - 1];
			}
			else {
				dst.at<Vec3b>(row, col) = Vec3b(0, 0, 0);
			}
		}
	}
	imshow("Final Result", dst);
#endif
#ifdef TEST35
	QString folderPath = "D:/test/";
	if (folderPath.isEmpty())
	{
		return 0;
	}
	QString file_path;
	QDir dir(folderPath);
	QStringList ImageList;
	ImageList << "*.bmp";//向字符串列表添加图片类型
	dir.setNameFilters(ImageList);//获得文件夹下图片的名字
	int ImageCount = dir.count();//获得dir里名字的个数，也表示文件夹下图片的个数
	for (int i = 0; i < ImageCount; i++)
	{
		Sleep(100);
		file_path = folderPath + "/" + dir[i];
		cv::Mat src = cv::imread(file_path.toStdString());

//		Mat src = imread("./test/X4.bmp");
		if (!src.data)
		{
			cout << "could not load image !";
			waitKey(0);
			return -1;
		}
		int threshold = 10;
		int vshadow = 200000;
		MEASURE_INFO measure_info;
		measure(src, threshold, vshadow, measure_info);
		Mat dst = measure_info.resultImg;
		cv::pyrDown(dst, dst, Size(dst.cols / 2, dst.rows / 2));
		cv::pyrDown(dst, dst, Size(dst.cols / 2, dst.rows / 2));
		QString str = QString("D:/test/result/%1.bmp").arg(i);
		cv::imwrite(str.toStdString(),dst);
	}

	
#endif
#ifdef TEST36
	Mat src = imread("./test/X2.bmp");
	if (!src.data)
	{
		cout << "could not load image !";
		waitKey(0);
		return -1;
	}
	CGetInnerRect *pCutContrlDeal = new CGetInnerRect();
	std::vector<cv::Point2f> defectLocation;
	float iangle;
	pCutContrlDeal->Execute(src, defectLocation, iangle);
	for (int i = 0; i < defectLocation.size(); i++)
	{
		cv::line(src, defectLocation[i], defectLocation[(i + 1) % 4], cv::Scalar(255, 0, 0), 2, 16);

	}
	waitKey(0);
#endif





#ifdef COURSE_02_TEST01
	Mat src = imread("QRCode.jpg");
	if (src.empty()) {
		printf("could not load image...\n");
		return -1;
	}
	Mat gray_src;
	cv::cvtColor(src, gray_src, COLOR_BGR2GRAY);
	Mat structureElement = getStructuringElement(MORPH_RECT, Size(5, 5), Point(-1, -1));
	// 	cv::erode(gray_src, gray_src, structureElement);//腐蚀
	// 	cv::dilate(gray_src, gray_src, structureElement, Point(-1, -1), 1);//膨胀
	Mat dst, norm_dst, normScaleDst;
	dst = Mat::zeros(gray_src.size(), CV_32FC1);

	int blockSize = 2;
	int ksize = 3;
	double k = 0.01;
	cv::cornerHarris(gray_src, dst, blockSize, ksize, k, BORDER_DEFAULT);
	cv::normalize(dst, norm_dst, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	//convertScaleAbs操作可实现图像增强等相关操作的快速运算
	cv::convertScaleAbs(norm_dst, normScaleDst);
	//	cv::threshold(normScaleDst, normScaleDst, 30, 255, THRESH_BINARY);
	int thresh = 20;
	Mat resultImg = src.clone();
	for (int row = 0; row < resultImg.rows; row++)
	{
		uchar* currentRow = normScaleDst.ptr(row);
		for (int col = 0; col < resultImg.cols; col++)
		{
			int value = (int)*currentRow;
			if (value > thresh)
			{
				circle(resultImg, Point(col, row), 2, Scalar(0, 0, 255), 2, 8, 0);
			}
			currentRow++;
		}
	}
#endif
#ifdef COURSE_02_TEST02
	Mat src = imread("QRCode.jpg");
	if (src.empty()) {
		std::printf("could not load image...\n");
		return -1;
	}
	Mat gray_src;
	cv::cvtColor(src, gray_src, COLOR_BGR2GRAY);
	cv::Mat bImg = gray_src.clone();
	// 	cv::blur(gray_src, bImg, Size(5, 5));
	int num_corners = 2500;
	int max_corners = 200;
	if (num_corners < 5)
	{
		num_corners = 5;
	}
	vector<Point2f> corners;
	double qualityLevel = 0.01;
	double minDistance = 1;
	int blockSize = 3;
	bool useHarris = false;
	double k = 0.04;

	Mat thImg;
	cv::threshold(bImg, thImg, 10, 255, THRESH_BINARY_INV);
	Mat structureElement = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
	cv::erode(thImg, thImg, structureElement);//腐蚀
	/*
	第一个参数是输入图像（8位或32位单通道图）。
	第二个参数是检测到的所有角点，类型为vector或数组，由实际给定的参数类型而定。如果是vector，那么它应该是一个包含cv::Point2f的vector对象；如果类型是cv::Mat,那么它的每一行对应一个角点，点的x、y位置分别是两列。
	第三个参数用于限定检测到的点数的最大值。
	第四个参数表示检测到的角点的质量水平（通常是0.10到0.01之间的数值，不能大于1.0）。
	第五个参数用于区分相邻两个角点的最小距离（小于这个距离得点将进行合并）。
	第六个参数是mask，如果指定，它的维度必须和输入图像一致，且在mask值为0处不进行角点检测。
	第七个参数是blockSize，表示在计算角点时参与运算的区域大小，常用值为3，但是如果图像的分辨率较高则可以考虑使用较大一点的值。
	第八个参数用于指定角点检测的方法，如果是true则使用Harris角点检测，false则使用Shi Tomasi算法。
	第九个参数是在使用Harris算法时使用，最好使用默认值0.04。
	*/
	cv::goodFeaturesToTrack(thImg, corners, num_corners, qualityLevel, minDistance, Mat(), blockSize, useHarris, k);
	std::printf("Number of Detected Corners:  %d\n", corners.size());

	Mat resultImg = gray_src.clone();
	cv::cvtColor(resultImg, resultImg, COLOR_GRAY2BGR);
	RNG rng(1234);
	for (size_t t = 0; t < corners.size(); t++)
	{
		circle(resultImg, corners[t], 2, Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), 2, 8, 0);
	}
#endif
#ifdef COURSE_02_TEST03
	src = imread("TM.jpg");
	if (src.empty()) {
		printf("could not load image...\n");
		return -1;
	}
	namedWindow("input image", CV_WINDOW_AUTOSIZE);
	imshow("input image", src);
	cvtColor(src, gray_src, COLOR_BGR2GRAY);
	// 计算特征值
	int blockSize = 3;
	int ksize = 3;
	double k = 0.04;
	harris_dst = Mat::zeros(src.size(), CV_32FC(6));
	harrisRspImg = Mat::zeros(src.size(), CV_32FC1);
	cornerEigenValsAndVecs(gray_src, harris_dst, blockSize, ksize, 4);
	// 计算响应
	for (int row = 0; row < harris_dst.rows; row++) {
		for (int col = 0; col < harris_dst.cols; col++) {
			double lambda1 = harris_dst.at<Vec6f>(row, col)[0];
			double lambda2 = harris_dst.at<Vec6f>(row, col)[1];
			harrisRspImg.at<float>(row, col) = lambda1*lambda2 - k*pow((lambda1 + lambda2), 2);
		}
	}
	minMaxLoc(harrisRspImg, &harris_min_rsp, &harris_max_rsp, 0, 0, Mat());
	namedWindow(harris_win, CV_WINDOW_AUTOSIZE);
	createTrackbar("Quality Value:", harris_win, &qualityLevel, max_count, CustomHarris_Demo);
	CustomHarris_Demo(0, 0);

	// 计算最小特征值
	shiTomasiRsp = Mat::zeros(src.size(), CV_32FC1);
	cornerMinEigenVal(gray_src, shiTomasiRsp, blockSize, ksize, 4);
	minMaxLoc(shiTomasiRsp, &shitomasi_min_rsp, &shitomasi_max_rsp, 0, 0, Mat());
	namedWindow(shitomasi_win, CV_WINDOW_AUTOSIZE);
	createTrackbar("Quality:", shitomasi_win, &sm_qualitylevel, max_count, CustomShiTomasi_Demo);
	CustomShiTomasi_Demo(0, 0);
#endif
#ifdef COURSE_02_TEST04
	src = imread("QRCode.jpg");
	if (src.empty()) {
		printf("could not load image...\n");
		return -1;
	}
	namedWindow("input image", CV_WINDOW_AUTOSIZE);
	imshow("input image", src);
	cvtColor(src, gray_src, COLOR_BGR2GRAY);
	namedWindow(output_title, CV_WINDOW_AUTOSIZE);
	createTrackbar("Corners:", output_title, &max_corners, max_count, SubPixel_Demo);
	SubPixel_Demo(0, 0);
#endif
#ifdef COURSE_02_TEST05
	Mat src = imread("TH.jpg", IMREAD_GRAYSCALE);
	if (src.empty()) {
		printf("could not load image...\n");
		return -1;
	}
	namedWindow("input image", WINDOW_AUTOSIZE);
	cv::imshow("input image", src);

	// SURF特征检测
	int minHessian = 10;
	Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create(minHessian);
	vector<KeyPoint> keypoints;
	detector->detect(src, keypoints, Mat());

	// 绘制关键点
	Mat keypoint_img;
	cv::drawKeypoints(src, keypoints, keypoint_img, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	cv::imshow("KeyPoints Image", keypoint_img);
#endif
#ifdef COURSE_02_TEST06
	Mat src = imread("QRCode.jpg", IMREAD_GRAYSCALE);
	if (src.empty()) {
		std::printf("could not load image...\n");
		return -1;
	}
	cv::imshow("input image", src);

	int numFeatures = 1000;
	Ptr<cv::xfeatures2d::SIFT> detector = cv::xfeatures2d::SIFT::create(numFeatures);
	vector<KeyPoint> keypoints;
	detector->detect(src, keypoints, Mat());
	std::printf("Total KeyPoints : %d\n", keypoints.size());

	Mat keypoint_img;
	cv::drawKeypoints(src, keypoints, keypoint_img, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	cv::imshow("SIFT KeyPoints", keypoint_img);

	cv::waitKey(0);
#endif
#ifdef COURSE_02_TEST07
	Mat src = imread("HU.jpg");
	if (src.empty()) {
		printf("could not load image...\n");
		return -1;
	}
	imshow("input image", src);

	HOGDescriptor hog = HOGDescriptor();
	hog.setSVMDetector(hog.getDefaultPeopleDetector());

	vector<Rect> foundLocations;
	hog.detectMultiScale(src, foundLocations,1, Size(8, 8), Size(32, 32), 1.05, 2);
	Mat result = src.clone();
	for (size_t t = 0; t < foundLocations.size(); t++) {
		rectangle(result, foundLocations[t], Scalar(0, 0, 255), 2, 8, 0);
	}
	imshow("HOG SVM Detector Demo", result);

	waitKey(0);
#endif
#ifdef COURSE_02_TEST08

	src = imread("FACE.jpg");
	if (src.empty()) {
		printf("could not load image...\n");
		return -1;
	}

	imshow("input image", src);

	// convert to gray
	cvtColor(src, gray_src, COLOR_BGR2GRAY);
	int width = gray_src.cols;
	int height = gray_src.rows;

	// 基本LBP演示
	Mat lbpImage = Mat::zeros(gray_src.rows - 2, gray_src.cols - 2, CV_8UC1);
	for (int row = 1; row < height - 1; row++) {
		for (int col = 1; col < width - 1; col++) {
			uchar c = gray_src.at<uchar>(row, col);
			uchar code = 0;
			code |= (gray_src.at<uchar>(row - 1, col - 1) > c) << 7;
			code |= (gray_src.at<uchar>(row - 1, col) > c) << 6;
			code |= (gray_src.at<uchar>(row - 1, col + 1) > c) << 5;
			code |= (gray_src.at<uchar>(row, col + 1) > c) << 4;
			code |= (gray_src.at<uchar>(row + 1, col + 1) > c) << 3;
			code |= (gray_src.at<uchar>(row + 1, col) > c) << 2;
			code |= (gray_src.at<uchar>(row + 1, col - 1) > c) << 1;
			code |= (gray_src.at<uchar>(row, col - 1) > c) << 0;
			lbpImage.at<uchar>(row - 1, col - 1) = code;
		}
	}
	imshow(output_tt, lbpImage);

	// ELBP 演示
	createTrackbar("ELBP Radius:", "ELBP Result", &current_radius, max_count, ELBP_Demo);
	ELBP_Demo(0, 0);

	waitKey(0);
#endif
#ifdef COURSE_02_TEST09
	Mat src = imread("FACE.jpg", IMREAD_GRAYSCALE);
	if (src.empty()) {
		printf("could not load image...\n");
		return -1;
	}
	namedWindow("input image", WINDOW_AUTOSIZE);
	imshow("input image", src);

	Mat sumii = Mat::zeros(src.rows + 1, src.cols + 1, CV_32FC1);
	Mat sqsumii = Mat::zeros(src.rows + 1, src.cols + 1, CV_64FC1);
	integral(src, sumii, sqsumii);

	Mat iiResult;
	normalize(sumii, iiResult, 0, 255, NORM_MINMAX, CV_8UC1, Mat());
	imshow("Integral Image", iiResult);
	waitKey(0);
#endif
	return a.exec();
}





/*********************************************************************************************************/
#ifdef TEST35
void measure(cv::Mat src, int threshold, int vshadow, MEASURE_INFO &measure_info)
{
	double t = cv::getTickCount();
	Mat src_gray;
	cv::cvtColor(src, src_gray, CV_BGR2GRAY);
	cv::blur(src_gray, src_gray, Size(11, 11));
	Mat threshold_output;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	Mat outPut;
	cv::threshold(src_gray, outPut, threshold, 255, THRESH_BINARY);
	Mat structureElement = getStructuringElement(MORPH_RECT, Size(21, 21), Point(-1, -1));
	cv::dilate(outPut, threshold_output, structureElement, Point(-1, -1), 1);
	cv::erode(threshold_output, threshold_output, structureElement);

	std::vector<std::vector<cv::Point>>  vvContour;
	cv::findContours(threshold_output, vvContour, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
	if (vvContour.size() == 0)
	{
		return;
	}
	int iMax = 0;
	double dMaxArea = cv::contourArea(vvContour[0]);
	for (int i = 1; i < vvContour.size(); i++)
	{
		double dA = cv::contourArea(vvContour[i]);
		if (dMaxArea < dA)
		{
			dMaxArea = dA;
			iMax = i;
		}
	}
	cv::Mat warpImg= threshold_output.clone();
	cv::RotatedRect rotRt = cv::minAreaRect(vvContour[iMax]);
	int a = rotRt.angle;
	if (abs(rotRt.angle) > 45)
	{
		rotRt.angle = 90 - abs(rotRt.angle);
	}

	cv::Mat RotateMat = cv::getRotationMatrix2D(rotRt.center, rotRt.angle, 1);
	warpAffine(threshold_output, warpImg, RotateMat, threshold_output.size());
	warpAffine(src, src, RotateMat, src.size());
	int vertical_shadow[IMAGE_SIZE::IAMGE_HEIGHT];
	int level_shadow[IMAGE_SIZE::IMAGE_WIDTH];
	memset(vertical_shadow, 0, sizeof(vertical_shadow));
	memset(level_shadow, 0, sizeof(level_shadow));
	for (int x = 0; x < warpImg.cols; x++)
	{
		for (int y = 0; y < warpImg.rows; y++)
		{
			uchar* ptr = warpImg.ptr<uchar>(y);
			vertical_shadow[x] += ptr[x];
		}
	}
	int index = 0;
	for (int x = 0; x < warpImg.cols; x++)
	{
		if (vertical_shadow[x] > vshadow)
		{
			vertical_shadow[x] = ++index;
		}
		else
		{
			index = 0;
			vertical_shadow[x] = index;
		}
	}
	int pt_x_s = 0;
	int pt_x_e = 0;
	bool inBlock = false;
	for (int x = 0; x < warpImg.cols; x++)
	{
		if (!inBlock &&vertical_shadow[x] != 0)
		{
			pt_x_s = x;
			inBlock = true;
		}
		if (inBlock &&vertical_shadow[x] == 0)
		{
			pt_x_e = x;
			inBlock = false;
		}
	}

	for (int y = 0; y < warpImg.rows; y++)
	{
		for (int x = 0; x < warpImg.cols; x++)
		{
			uchar* ptr = warpImg.ptr<uchar>(y);
			level_shadow[y] += ptr[x];
		}
	}
	int index2 = 0;
	for (int y = 0; y < warpImg.rows; y++)
	{
		if (level_shadow[y] > vshadow)
		{
			level_shadow[y] = ++index2;
		}
		else
		{
			index2 = 0;
			level_shadow[y] = index2;
		}
	}
	int pt_y_s = 0;
	int pt_y_e = 0;
	bool inBlock2 = false;
	for (int y = 0; y < warpImg.rows; y++)
	{
		if (!inBlock2 &&level_shadow[y] != 0)
		{
			pt_y_s = y;
			inBlock2 = true;
		}
		if (inBlock2 &&level_shadow[y] == 0)
		{
			pt_y_e = y;
			inBlock2 = false;
		}
	}
	Mat clImg;
	src.copyTo(clImg);
	cv::Point p1(pt_y_s, pt_x_s);
	cv::Point p2(pt_y_e, pt_x_e);
	Rect rect(pt_x_s, pt_y_s, pt_x_e - pt_x_s, pt_y_e - pt_y_s);
	cv::rectangle(clImg, rect, Scalar(0, 0, 255), 2, 8, 0);
	int width = pt_x_e - pt_x_s;
	double time = (getTickCount() - t) / getTickFrequency();
	char tag[256];
	sprintf(tag, "width=%d,time=%f ms", width, time*1000);
	cv::putText(clImg, tag, cv::Point2i(pt_y_s, pt_x_s - 200), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(255, 0, 0), 3, 8, false);
	
	measure_info.resultImg = clImg.clone();
	measure_info.rect = rect;
	measure_info.time = time;
	measure_info.width = width;
}
#endif
#ifdef COURSE_02_TEST03
void CustomHarris_Demo(int, void*) 
{
	if (qualityLevel < 10)
	{
		qualityLevel = 10;
	}
	Mat resultImg = src.clone();
	float t = harris_min_rsp + (((double)qualityLevel) / max_count)*(harris_max_rsp - harris_min_rsp);
	for (int row = 0; row < src.rows; row++)
	{
		for (int col = 0; col < src.cols; col++)
		{
			float v = harrisRspImg.at<float>(row, col);
			if (v > t) 
			{
				circle(resultImg, Point(col, row), 2, Scalar(0, 0, 255), 2, 8, 0);
			}
		}
	}
	imshow(harris_win, resultImg);
}

void CustomShiTomasi_Demo(int, void*) 
{
	if (sm_qualitylevel < 20) 
	{
		sm_qualitylevel = 20;
	}

	Mat resultImg = src.clone();
	float t = shitomasi_min_rsp + (((double)sm_qualitylevel) / max_count)*(shitomasi_max_rsp - shitomasi_min_rsp);
	for (int row = 0; row < src.rows; row++) 
	{
		for (int col = 0; col < src.cols; col++) 
		{
			float v = shiTomasiRsp.at<float>(row, col);
			if (v > t) 
			{
				circle(resultImg, Point(col, row), 2, Scalar(0, 0, 255), 2, 8, 0);
			}
		}
	}
	imshow(shitomasi_win, resultImg);
}
#endif

#ifdef COURSE_02_TEST04
void SubPixel_Demo(int, void*) {
	if (max_corners < 5) {
		max_corners = 5;
	}
	vector<Point2f> corners;
	double qualityLevel = 0.01;
	double minDistance = 10;
	int blockSize = 3;
	double k = 0.04;
	goodFeaturesToTrack(gray_src, corners, max_corners, qualityLevel, minDistance, Mat(), blockSize, false, k);
	cout << "number of corners: " << corners.size() << endl;
	Mat resultImg = src.clone();
	for (size_t t = 0; t < corners.size(); t++) {
		circle(resultImg, corners[t], 2, Scalar(0, 0, 255), 2, 8, 0);
	}
	imshow(output_title, resultImg);
	Size winSize = Size(5, 5);
	Size zerozone = Size(-1, -1);
	TermCriteria tc = TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 40, 0.001);
	cornerSubPix(gray_src, corners, winSize, zerozone, tc);

	for (size_t t = 0; t < corners.size(); t++) {
		cout << (t + 1) << " .point[x, y] = " << corners[t].x << " , " << corners[t].y << endl;
	}
}
#endif
#ifdef COURSE_02_TEST08
void ELBP_Demo(int, void*) {
	int offset = current_radius * 2;
	Mat elbpImage = Mat::zeros(gray_src.rows - offset, gray_src.cols - offset, CV_8UC1);
	int width = gray_src.cols;
	int height = gray_src.rows;

	int numNeighbors = 8;
	for (int n = 0; n < numNeighbors; n++) {
		float x = static_cast<float>(current_radius) * cos(2.0 * CV_PI*n / static_cast<float>(numNeighbors));
		float y = static_cast<float>(current_radius) * -sin(2.0 * CV_PI*n / static_cast<float>(numNeighbors));

		int fx = static_cast<int>(floor(x));
		int fy = static_cast<int>(floor(y));
		int cx = static_cast<int>(ceil(x));
		int cy = static_cast<int>(ceil(y));

		float ty = y - fy;
		float tx = x - fx;

		float w1 = (1 - tx)*(1 - ty);
		float w2 = tx*(1 - ty);
		float w3 = (1 - tx)* ty;
		float w4 = tx*ty;

		for (int row = current_radius; row < (height - current_radius); row++) {
			for (int col = current_radius; col < (width - current_radius); col++) {
				float t = w1* gray_src.at<uchar>(row + fy, col + fx) + w2* gray_src.at<uchar>(row + fy, col + cx) +
					w3* gray_src.at<uchar>(row + cy, col + fx) + w4* gray_src.at<uchar>(row + cy, col + cx);
				elbpImage.at<uchar>(row - current_radius, col - current_radius) +=
					((t > gray_src.at<uchar>(row, col)) && (abs(t - gray_src.at<uchar>(row, col)) > std::numeric_limits<float>::epsilon())) << n;
			}
		}
	}
	imshow(output_et, elbpImage);
	return;
}
#endif