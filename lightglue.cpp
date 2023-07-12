#include"lightglue.h"


LightGlue::LightGlue(int wid, int hei)
{
	this->m_width = wid;
	this->m_height = hei;

	/*	HMODULE g_hInstance;
		g_hInstance = ::GetCurrentModule();
		HRSRC hRcmodel = FindResource(g_hInstance, MAKEINTRESOURCE(IDR_LIGHTGLUE1), "LIGHTGLUE");
		DWORD dwModelSize = SizeofResource(g_hInstance, hRcmodel);
		HGLOBAL hModelGlobal = LoadResource(g_hInstance, hRcmodel);
		void* pModelBuffer = LockResource(hModelGlobal);

		Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "LightGlue");
		Ort::SessionOptions sessionOptions;
		sessionOptions.SetIntraOpNumThreads(1);
		sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
		this->m_extractorSession = std::shared_ptr<Ort::Session>(new Ort::Session(env, (const void*)pModelBuffer, dwModelSize, sessionOptions));*/
}
void LightGlue::match(const ImageFeatures& features1, const ImageFeatures& features2,
	MatchesInfo& matches_info)
{
	Ort::Env env(ORT_LOGGING_LEVEL_FATAL, "LightGlue");
	Ort::SessionOptions sessionOptions;
	sessionOptions.SetIntraOpNumThreads(1);
	sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
	std::wstring wstr = L"D:\\superpoint_lightglue.onnx";
	// Load the LightGlue network
	static Ort::Session lightglueSession(env, wstr.c_str(), sessionOptions);
	vector<float>kp1; kp1.resize(features1.keypoints.size() * 2);
	vector<float>kp2; kp2.resize(features2.keypoints.size() * 2);
	float halfwid = this->m_width / 2;
	float halfhei = this->m_height / 2;
	for (int i = 0; i < features1.keypoints.size(); i++)
	{
		kp1[2 * i] = (features1.keypoints[i].pt.x - halfwid) / halfwid;
		kp1[2 * i + 1] = (features1.keypoints[i].pt.y - halfhei) / halfhei;
	}
	for (int i = 0; i < features2.keypoints.size(); i++)
	{
		kp2[2 * i] = (features2.keypoints[i].pt.x - halfwid) / halfwid;
		kp2[2 * i + 1] = (features2.keypoints[i].pt.y - halfhei) / halfhei;
	}

	vector<float>des1; des1.resize(features1.keypoints.size() * 256);
	Mat des1mat = features1.descriptors.getMat(ACCESS_READ);
	for (int w = 0; w < des1mat.cols; w++)
	{
		for (int h = 0; h < des1mat.rows; h++)
		{
			int index = h * features1.descriptors.cols + w;
			des1[index] = des1mat.at<float>(h, w);
		}
	}
	vector<float>des2; des2.resize(features2.keypoints.size() * 256);
	Mat des2mat = features2.descriptors.getMat(ACCESS_READ);
	for (int w = 0; w < des2mat.cols; w++)
	{
		for (int h = 0; h < des2mat.rows; h++)
		{
			int index = h * features2.descriptors.cols + w;
			des2[index] = des2mat.at<float>(h, w);
		}
	}

	const char* input_names[] = { "kpts0", "kpts1", "desc0", "desc1" };
	Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

	std::vector<Ort::Value> inputTensor;
	vector<int64_t> kp1Shape{ 1,(int64_t)features1.keypoints.size(), 2 };
	inputTensor.emplace_back(Ort::Value::CreateTensor<float>(memoryInfo, kp1.data(), kp1.size(), kp1Shape.data(), kp1Shape.size()));
	vector<int64_t> kp2Shape{ 1,(int64_t)features2.keypoints.size(), 2 };
	inputTensor.emplace_back(Ort::Value::CreateTensor<float>(memoryInfo, kp2.data(), kp2.size(), kp2Shape.data(), kp2Shape.size()));
	vector<int64_t> des1Shape{ 1,(int64_t)features1.keypoints.size(), features1.descriptors.cols };
	inputTensor.emplace_back(Ort::Value::CreateTensor<float>(memoryInfo, des1.data(), des1.size(), des1Shape.data(), des1Shape.size()));
	vector<int64_t> des2Shape{ 1,(int64_t)features2.keypoints.size(), features2.descriptors.cols };
	inputTensor.emplace_back(Ort::Value::CreateTensor<float>(memoryInfo, des2.data(), des2.size(), des2Shape.data(), des2Shape.size()));
	const char* output_names[] = { "matches0","matches1","mscores0","mscores1" };
	Ort::RunOptions run_options;
	vector<Ort::Value> outputs = lightglueSession.Run(run_options, input_names, inputTensor.data(), 4, output_names, 4);
	std::vector<int64_t> match1shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
	int64_t* match1 = (int64_t*)outputs[0].GetTensorMutableData<void>();
	int match1counts = match1shape[1];

	std::vector<int64_t> mscoreshape = outputs[3].GetTensorTypeAndShapeInfo().GetShape();
	float* mscore1 = (float*)outputs[3].GetTensorMutableData<void>();
	int mscorecounts = mscoreshape[1];

	std::vector<int64_t> match2shape = outputs[1].GetTensorTypeAndShapeInfo().GetShape();
	int64_t* match2 = (int64_t*)outputs[1].GetTensorMutableData<void>();
	int match2counts = match2shape[1];

	vector<float>matchs;
	for (int i = 0; i < match1counts; i++)
	{
		if (match1[i] > -1 && mscore1[i] > 0.0f && match2[match1[i]] == i)
		{
			matchs.push_back(i);
			matchs.push_back(match1[i]);
		}
	}
	std::cout << "matches count:" << matchs.size() << std::endl;
	// Construct point-point correspondences for transform estimation
	Mat src_points(1, static_cast<int>(matchs.size() / 2), CV_32FC2);
	Mat dst_points(1, static_cast<int>(matchs.size() / 2), CV_32FC2);
	for (size_t i = 0; i < matchs.size() / 2; ++i)
	{
		src_points.at<Point2f>(0, static_cast<int>(i)) = features1.keypoints[matchs[2 * i]].pt;
		dst_points.at<Point2f>(0, static_cast<int>(i)) = features2.keypoints[matchs[2 * i + 1]].pt;
	}

	// Find pair-wise motion
	matches_info.H = estimateAffine2D(src_points, dst_points, matches_info.inliers_mask);

	if (matches_info.H.empty()) {
		// could not find transformation
		matches_info.confidence = 0;
		matches_info.num_inliers = 0;
		return;
	}

	// Find number of inliers
	matches_info.num_inliers = 0;
	for (size_t i = 0; i < matches_info.inliers_mask.size(); ++i)
		if (matches_info.inliers_mask[i])
			matches_info.num_inliers++;

	// These coeffs are from paper M. Brown and D. Lowe. "Automatic Panoramic
	// Image Stitching using Invariant Features"
	matches_info.confidence =
		matches_info.num_inliers / (8 + 0.3 * matches_info.matches.size());

	/* should we remove matches between too close images? */
	// matches_info.confidence = matches_info.confidence > 3. ? 0. : matches_info.confidence;

	// extend H to represent linear transformation in homogeneous coordinates
	matches_info.H.push_back(Mat::zeros(1, 3, CV_64F));
	matches_info.H.at<double>(2, 2) = 1;

}

