#pragma once
#include <vector>
#include <string>
#include <memory>
#include <Windows.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/stitching.hpp"
#include<iostream>
#include <fstream> //for file operations
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/core/ocl.hpp"
#include <opencv2/imgproc.hpp>
#include "opencv2/calib3d.hpp"
#include <onnxruntime_cxx_api.h>
#include <math.h>
#include "common.h"
using namespace cv::detail;
using namespace cv;
using namespace std;

class  FEATURE_MATCHER_EXPORTS LightGlue :public FeaturesMatcher
{
protected:

	CV_WRAP_AS(apply) void operator ()(const ImageFeatures& features1, const ImageFeatures& features2,
		CV_OUT MatchesInfo& matches_info) {
		match(features1, features2, matches_info);
	}
public:
	LightGlue();
	void match(const ImageFeatures& features1, const ImageFeatures& features2,
		MatchesInfo& matches_info);

};
