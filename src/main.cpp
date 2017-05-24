#include <opencv2/opencv.hpp>
#include <iostream>

cv::Mat flowToBGR(const cv::Mat& flow)
{
  std::vector<cv::Mat> vflow;
  cv::Mat _hsv[3], hsv;
  cv::Mat magnitude, angle;
  cv::Mat flow_bgr;
  cv::split(flow, vflow);
  cv::cartToPolar(vflow[0], vflow[1], magnitude, angle, true);
  
  cv::normalize(magnitude, magnitude, 0, 1, cv::NORM_MINMAX);
  // magnitude *= 255;
  // cv::normalize(magnitude, magnitude, 0, 255, cv::NORM_MINMAX);
  
  _hsv[0] = angle;
  _hsv[1] = cv::Mat::ones(angle.size(), CV_32F);      
  _hsv[2] = magnitude;

  cv::merge(_hsv, 3, hsv);

  cv::cvtColor(hsv, flow_bgr, cv::COLOR_HSV2BGR);
  flow_bgr.convertTo(flow_bgr, CV_8UC3, 255.0);
    
  return flow_bgr;
}

cv::Rect maxRoiContour(const cv::Mat& flow_gray)
{
  
  std::vector<std::vector<cv::Point> > contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(flow_gray, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
      
  cv::Rect roi(0, 0, 0, 0);

  if (!contours.empty())
    {
      int idx_largest_contour = -1;
      double area_largest_contour = 0.0;

      for (int i = 0; i < contours.size(); ++i)
	{
	  double area = cv::contourArea(contours[i]);
	  if (area_largest_contour < area)
	    {
	      area_largest_contour = area;
	      idx_largest_contour = i;
	    }
	}

      if (area_largest_contour > 500)
	{
	  roi = cv::boundingRect(contours[idx_largest_contour]);
	}
    }
  
  return roi;
      
}

int main()
{
  cv::VideoCapture cap(0);
  if (!cap.isOpened())
    return -1;
  
  cv::Mat prevFrame, curFrame, prevGray, curGray;
  cv::Mat flow;
  cv::Mat flow_gray;
  cv::Mat kernel;
  kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
  
  cap >> prevFrame;
  cv::cvtColor(prevFrame, prevGray, cv::COLOR_BGR2GRAY);
  cv::GaussianBlur(prevGray, prevGray, cv::Size(5, 5), 0, 0);
  cv::equalizeHist(prevGray, prevGray);

  while(1)
    {
      cap >> curFrame;
      cv::cvtColor(curFrame, curGray, cv::COLOR_BGR2GRAY);
      cv::GaussianBlur(curGray, curGray, cv::Size(5, 5), 0, 0);
      cv::equalizeHist(curGray, curGray);
      
      cv::calcOpticalFlowFarneback(prevGray, curGray, flow, 0.5, 5, 21, 3, 5, 1.2, cv::OPTFLOW_FARNEBACK_GAUSSIAN); 

      cv::cvtColor(flowToBGR(flow), flow_gray, cv::COLOR_BGR2GRAY);

      cv::Mat thresh;

      cv::threshold(flow_gray, flow_gray, 25, 255, CV_THRESH_BINARY);
      cv::Mat dilateElement = cv::getStructuringElement( cv::MORPH_RECT, cv::Size(10,10));
      cv::dilate(flow_gray, flow_gray, dilateElement);
      
      cv::Rect roi = maxRoiContour(flow_gray);

      if (roi.area() > 0)
	cv::rectangle(curFrame, roi, cv::Scalar(0, 255, 0), 4);	
      cv::imshow("curFrame", curFrame);
      
      curGray.copyTo(prevGray);
      if (cv::waitKey(1) == 113)
	break;
    }
  
  cv::destroyAllWindows();  
  return 0;
}
