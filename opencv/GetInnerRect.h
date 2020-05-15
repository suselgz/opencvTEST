#pragma once
#include "stdafx.h"
class CGetInnerRect
{
public:
	CGetInnerRect();
	~CGetInnerRect();

	int Execute(cv::Mat& src, std::vector<cv::Point2f>& vecInnerVertex, float &angle);

private:
	void GetVertex(std::vector<cv::Point>& vecContour, std::vector<cv::Point2f>& vecRect, std::vector<int>& vecNNIdx);
};

