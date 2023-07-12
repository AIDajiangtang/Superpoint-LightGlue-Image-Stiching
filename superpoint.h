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

class  FEATURE_MATCHER_EXPORTS SuperPoint :public Feature2D
{
protected:
	vector<float> ApplyTransform(const Mat& image, float& mean, float& std);
public:
	SuperPoint();
	virtual void detectAndCompute(InputArray image, InputArray mask,
		std::vector<KeyPoint>& keypoints,
		OutputArray descriptors,
		bool useProvidedKeypoints = false);
	virtual void detect(InputArray image,
		std::vector<KeyPoint>& keypoints,
		InputArray mask = noArray());
	virtual void compute(InputArray image,
		std::vector<KeyPoint>& keypoints,
		OutputArray descriptors);

};