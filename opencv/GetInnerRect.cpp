#include "GetInnerRect.h"

CGetInnerRect::CGetInnerRect()
{
}


CGetInnerRect::~CGetInnerRect()
{
}

int GetBound(std::vector<cv::Point2f>& vec, cv::Point pt1, cv::Point pt2, int mark, int iRows, int iCols)
{
	cv::Rect rt;
	if (mark==0) //上
	{
		rt.x = std::min(pt1.x, pt2.x) + 20;
		rt.width = abs(pt1.x - pt2.x) - 40;
		rt.y = int((float(pt1.y) + float(pt2.y))*0.5f) - 20;
		rt.height = 40;

		int iY = 0;
		for (int i=0; i<vec.size(); i++)
		{
			if (rt.contains(vec[i]))
			{
				iY = std::max(iY, int(vec[i].y));
			}
		}
		return iY;
	}
	else if (mark == 1) //下
	{
		rt.x = std::min(pt1.x, pt2.x) + 20;
		rt.width = abs(pt1.x - pt2.x) - 40;
		rt.y = int((float(pt1.y) + float(pt2.y))*0.5f) - 20;
		rt.height = 40;

		int iY = iRows;
		for (int i = 0; i < vec.size(); i++)
		{
			if (rt.contains(vec[i]))
			{
				iY = std::min(iY, int(vec[i].y));
			}
		}
		return iY;
	}
	else if (mark == 2)//左
	{
		rt.y = std::min(pt1.y, pt2.y) + 20;
		rt.height = abs(pt1.y - pt2.y) - 40;
		rt.x = int((float(pt1.x) + float(pt2.x))*0.5f) - 20;
		rt.width = 40;

		int iX = 0;
		for (int i = 0; i < vec.size(); i++)
		{
			if (rt.contains(vec[i]))
			{
				iX = std::max(iX, int(vec[i].x));
			}
		}
		return iX;
	}
	else
	{
		rt.y = std::min(pt1.y, pt2.y) + 20;
		rt.height = abs(pt1.y - pt2.y) - 40;
		rt.x = int((float(pt1.x) + float(pt2.x))*0.5f) - 20;
		rt.width = 40;

		int iX = iCols;
		for (int i = 0; i < vec.size(); i++)
		{
			if (rt.contains(vec[i]))
			{
				iX = std::min(iX, int(vec[i].x));
			}
		}
		return iX;
	}
}

int CGetInnerRect::Execute(cv::Mat& src, std::vector<cv::Point2f>& vecInnerVertex, float &angle)
{
	cv::Mat grayimg;
	if (src.channels()==3)
	{
		cv::cvtColor(src, grayimg, cv::COLOR_BGR2GRAY);
	}
	else if (src.channels()==4)
	{
		cv::cvtColor(src, grayimg, cv::COLOR_BGRA2GRAY);
	}
	else
	{
		grayimg = src;
	}

	cv::Mat binaryImg;
	cv::threshold(grayimg, binaryImg, 20, 0xff, cv::THRESH_BINARY);
	cv::resize(binaryImg, binaryImg, cv::Size(binaryImg.cols >> 1, binaryImg.rows >> 1), 0, 0, cv::INTER_NEAREST);
	cv::erode(binaryImg, binaryImg, cv::Mat::ones(3, 3, CV_8U));
	cv::dilate(binaryImg, binaryImg, cv::Mat::ones(3, 3, CV_8U));
	cv::dilate(binaryImg, binaryImg, cv::Mat::ones(3, 3, CV_8U));
	//cv::dilate(binaryImg, binaryImg, cv::Mat::ones(3, 3, CV_8U));
	//cv::erode(binaryImg, binaryImg, cv::Mat::ones(3, 3, CV_8U));
	cv::erode(binaryImg, binaryImg, cv::Mat::ones(3, 3, CV_8U));

	//cv::Mat tempMat = binaryImg.clone();
	std::vector<std::vector<cv::Point>>  vvContour;
	cv::findContours(binaryImg, vvContour, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
	if (vvContour.size()==0)
	{
		return 1;
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
	cv::RotatedRect rotRt = cv::minAreaRect(vvContour[iMax]);

	//cv::Point2f srcCenter(binaryImg.cols >> 1, binaryImg.rows >> 1);
	cv::Mat RotateMat = cv::getRotationMatrix2D(rotRt.center, rotRt.angle, 1);


	std::vector<cv::Point2f> vecDst, vecSrc;
	for (auto&& v : vvContour[iMax])
	{
		vecSrc.push_back(cv::Point2f(v.x, v.y));
	}
	std::vector<cv::Point2f> vecPoint(4);
	rotRt.points(&vecPoint[0]);
	std::vector<int> vecNNIdx;
	GetVertex(vvContour[iMax], vecPoint, vecNNIdx);

	cv::transform(vecSrc, vecDst, RotateMat);
	cv::Rect tempRt = cv::boundingRect(vecDst);
	cv::Point2f center = cv::Point2f(tempRt.x + tempRt.width*0.5f, tempRt.y + tempRt.height*0.5f);
	binaryImg.setTo(0x00);
	for (int i = 0; i < vecDst.size(); i++)
	{
		int idx = int(vecDst[i].y) * binaryImg.cols + int(vecDst[i].x);
		binaryImg.data[idx] = 255;
	}

	//int iIndex[4] = {0};
	for (int i = 0; i < vecNNIdx.size(); i++)
	{
		cv::Point2f diff = vecDst[vecNNIdx[i]] - center;
		float f = std::atan2(diff.y, diff.x);
		if (f > 0 && f < CV_PI*0.5f)//四象限
		{
			vecPoint[2] = vecDst[vecNNIdx[i]];
			//iIndex[2] = vecNNIdx[i];
		}
		else if (f > CV_PI*0.5f)//三象限
		{
			vecPoint[3] = vecDst[vecNNIdx[i]];
			//iIndex[3] = vecNNIdx[i];
		}
		else if (f < 0 && f < -CV_PI*0.5f)//二象限
		{
			vecPoint[0] = vecDst[vecNNIdx[i]];
			//iIndex[0] = vecNNIdx[i];
		}
		else
		{
			vecPoint[1] = vecDst[vecNNIdx[i]];//一象限
			//iIndex[1] = vecNNIdx[i];
		}
	}
	//top
	cv::Rect rr;
	rr.y = GetBound(vecDst, vecPoint[0], vecPoint[1], 0, binaryImg.rows, binaryImg.cols);
	rr.x = GetBound(vecDst, vecPoint[0], vecPoint[3], 2, binaryImg.rows, binaryImg.cols);
	rr.height = GetBound(vecDst, vecPoint[2], vecPoint[3], 1, binaryImg.rows, binaryImg.cols);
	rr.width = GetBound(vecDst, vecPoint[2], vecPoint[1], 3, binaryImg.rows, binaryImg.cols);
// 	rr.width = rr.width - rr.x + 1;
// 	rr.height = rr.height - rr.y + 1;
	std::vector<cv::Point2f> vecInnerRt(4);
	vecInnerRt[0].x = rr.x;
	vecInnerRt[0].y = rr.y;
	vecInnerRt[1].x = rr.width;
	vecInnerRt[1].y = rr.y;
	vecInnerRt[2].x = rr.width;
	vecInnerRt[2].y = rr.height;
	vecInnerRt[3].x = rr.x;
	vecInnerRt[3].y = rr.height;
	angle = -rotRt.angle;
	RotateMat = cv::getRotationMatrix2D(rotRt.center, -rotRt.angle, 1);
	cv::transform(vecInnerRt, vecInnerVertex, RotateMat);
	for (int i = 0; i < vecInnerVertex.size(); i++)
	{
		vecInnerVertex[i].x *= 2.0f;
		vecInnerVertex[i].y *= 2.0f;
	}

	for (auto&& v : vvContour)
	{
		v.clear();
	}
	vvContour.clear();
	vecSrc.clear();
	vecDst.clear();
	vecNNIdx.clear();
#ifdef _DEBUG
	cv::Mat rgbImg;
	cv::cvtColor(grayimg, rgbImg, cv::COLOR_GRAY2BGR);
	
	
	//cv::rectangle(rgbImg, rr, cv::Scalar(0,0,255));
	cv::line(rgbImg, cv::Point(vecInnerVertex[0].x, vecInnerVertex[0].y), cv::Point(vecInnerVertex[1].x, vecInnerVertex[1].y), cv::Scalar(0, 255, 0));
	cv::line(rgbImg, cv::Point(vecInnerVertex[2].x, vecInnerVertex[2].y), cv::Point(vecInnerVertex[1].x, vecInnerVertex[1].y), cv::Scalar(0, 255, 0));
	cv::line(rgbImg, cv::Point(vecInnerVertex[0].x, vecInnerVertex[0].y), cv::Point(vecInnerVertex[3].x, vecInnerVertex[3].y), cv::Scalar(0, 255, 0));
	cv::line(rgbImg, cv::Point(vecInnerVertex[2].x, vecInnerVertex[2].y), cv::Point(vecInnerVertex[3].x, vecInnerVertex[3].y), cv::Scalar(0, 255, 0));
// 	for (int i = 0; i < vecDst.size(); i++)
// 	{
// 		int idx = int(vecDst[i].y) * rgbImg.cols + int(vecDst[i].x);
// 		idx = idx * 3;
// 		rgbImg.data[idx] = 255;
// 		rgbImg.data[idx+1] = 0;
// 		rgbImg.data[idx+2] = 255;
// 	}
// 	cv::circle(rgbImg, vvContour[iMax][vecNNIdx[0]], 3, cv::Scalar(255, 255, 128));
// 	cv::circle(rgbImg, vvContour[iMax][vecNNIdx[1]], 3, cv::Scalar(255, 255, 128));
// 	cv::circle(rgbImg, vvContour[iMax][vecNNIdx[2]], 3, cv::Scalar(255, 255, 128));
// 	cv::circle(rgbImg, vvContour[iMax][vecNNIdx[3]], 3, cv::Scalar(255, 255, 128));
	src = rgbImg.clone();
#endif

	return 0;
}

void CGetInnerRect::GetVertex(std::vector<cv::Point>& vecContour, std::vector<cv::Point2f>& vecRect, std::vector<int>& vecNNIdx)
{
	cv::Mat srcMat = cv::Mat(vecContour).reshape(1);
	cv::Mat tempMat;
	srcMat.convertTo(tempMat, CV_32F);
	cv::flann::KDTreeIndexParams knnParam(2);
	cv::flann::Index knn(tempMat, knnParam);
	unsigned querNum = 1;
	std::vector<float> vecQuery(2);
	std::vector<int> vecIndex(querNum);
	std::vector<float> vecDist(querNum);
	cv::flann::SearchParams params(32);

	vecNNIdx.clear();
	int iMaxRlt = 100;
	for (int i = 0; i < 4; i++)
	{
		vecQuery[0] = vecRect[i].x;
		vecQuery[1] = vecRect[i].y;
		knn.knnSearch(vecQuery, vecIndex, vecDist, querNum);
// 		double dRadius = sqrt((vecQuery[0]-vecContour[vecIndex[0]].x)*(vecQuery[0] - vecContour[vecIndex[0]].x) + (vecQuery[1] - vecContour[vecIndex[0]].y)*(vecQuery[1] - vecContour[vecIndex[0]].y));
// 		knn.radiusSearch(vecQuery, vecIndex, vecDist, dRadius, iMaxRlt);
// 		vecNNIdx.insert(vecNNIdx.end(), vecIndex.begin(), vecIndex.end());
		vecNNIdx.push_back(vecIndex[0]);
	}
}
