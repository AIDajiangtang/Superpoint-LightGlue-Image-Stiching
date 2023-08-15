#include"superpoint.h"
#include <onnxruntime_cxx_api.h>

SuperPoint::SuperPoint(std::wstring modelPath)
{
	this->m_modelPath = modelPath;
	/*HMODULE g_hInstance;
	g_hInstance = ::GetCurrentModule();
	HRSRC hRcmodel = FindResource(g_hInstance, MAKEINTRESOURCE(IDR_SUPERPOINT1), "SUPERPOINT");
	DWORD dwModelSize = SizeofResource(g_hInstance, hRcmodel);
	HGLOBAL hModelGlobal = LoadResource(g_hInstance, hRcmodel);
	void* pModelBuffer = LockResource(hModelGlobal);

	Ort::Env env(ORT_LOGGING_LEVEL_FATAL, "SuperPoint");
	Ort::SessionOptions sessionOptions;
	sessionOptions.SetIntraOpNumThreads(1);
	sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
	this->m_extractorSession = std::shared_ptr<Ort::Session>(new Ort::Session(env, (const void*)pModelBuffer, dwModelSize, sessionOptions));*/
}
vector<float> SuperPoint::ApplyTransform(const Mat& image, float& mean, float& std)
{
	Mat resized, floatImage;
	image.convertTo(floatImage, CV_32FC1);
	//mean = 0.0f;
	//std = 0.0f;
	//Scalar meanScalar, stdScalar;
	//meanStdDev(floatImage, meanScalar, stdScalar);
	//mean = static_cast<float>(meanScalar.val[0]);
	//std = static_cast<float>(stdScalar.val[0]);

	vector<float> imgData;
	for (int h = 0; h < image.rows; h++)
	{
		for (int w = 0; w < image.cols; w++)
		{
			/*imgData.push_back((floatImage.at<float>(h, w) - mean) / std);*/
			imgData.push_back(floatImage.at<float>(h, w) / 255.0f);
		}
	}
	return imgData;
}
void SuperPoint::detectAndCompute(InputArray image, InputArray mask,
	std::vector<KeyPoint>& keypoints,
	OutputArray descriptors,
	bool useProvidedKeypoints)
{
	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "SuperPoint");
	Ort::SessionOptions sessionOptions;
	sessionOptions.SetIntraOpNumThreads(1);
	sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
	static Ort::Session extractorSession(env, this->m_modelPath.c_str(), sessionOptions);

	Mat img = image.getMat();
	Mat grayImg;
	cvtColor(img, grayImg, COLOR_BGR2GRAY);
	float mean, std;
	vector<float> imgData = ApplyTransform(grayImg, mean, std);

	vector<int64_t> inputShape{ 1, 1, grayImg.rows, grayImg.cols };

	Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, imgData.data(), imgData.size(), inputShape.data(), inputShape.size());

	const char* input_names[] = { "image" };
	const char* output_names[] = { "keypoints","scores","descriptors" };
	Ort::RunOptions run_options;
	vector<Ort::Value> outputs = extractorSession.Run(run_options, input_names, &inputTensor, 1, output_names, 3);

	std::vector<int64_t> kpshape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
	int64* kp = (int64*)outputs[0].GetTensorMutableData<void>();
	int keypntcounts = kpshape[1];
	keypoints.resize(keypntcounts);
	for (int i = 0; i < keypntcounts; i++)
	{
		KeyPoint p;
		int index = i * 2;
		p.pt.x = (float)kp[index];
		p.pt.y = (float)kp[index + 1];
		keypoints[i] = p;
	}

	std::vector<int64_t> desshape = outputs[2].GetTensorTypeAndShapeInfo().GetShape();
	float* des = (float*)outputs[2].GetTensorMutableData<void>();

	Mat desmat = descriptors.getMat();
	desmat.create(Size(desshape[2], desshape[1]), CV_32FC1);
	for (int h = 0; h < desshape[1]; h++)
	{
		for (int w = 0; w < desshape[2]; w++)
		{
			int index = h * desshape[2] + w;
			desmat.at<float>(h, w) = des[index];
		}
	}
	desmat.copyTo(descriptors);

}
void SuperPoint::detect(InputArray image,
	std::vector<KeyPoint>& keypoints,
	InputArray mask)
{
	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "SuperPoint");
	Ort::SessionOptions sessionOptions;
	sessionOptions.SetIntraOpNumThreads(1);
	sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
	static Ort::Session extractorSession(env, this->m_modelPath.c_str(), sessionOptions);

	Mat img = image.getMat();
	Mat grayImg;
	cvtColor(img, grayImg, COLOR_BGR2GRAY);
	float mean, std;
	vector<float> imgData = ApplyTransform(grayImg, mean, std);
	vector<int64_t> inputShape{ 1, 1, grayImg.rows, grayImg.cols };
	Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, imgData.data(), imgData.size(), inputShape.data(), inputShape.size());

	const char* input_names[] = { "image" };
	const char* output_names[] = { "keypoints","scores","descriptors" };
	Ort::RunOptions run_options;
	vector<Ort::Value> outputs = extractorSession.Run(run_options, input_names, &inputTensor, 1, output_names, 3);

	std::vector<int64_t> kpshape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
	int64* kp = (int64*)outputs[0].GetTensorMutableData<void>();
	int keypntcounts = kpshape[1];
	keypoints.resize(keypntcounts);
	for (int i = 0; i < keypntcounts; i++)
	{
		KeyPoint p;
		int index = i * 2;
		p.pt.x = (float)kp[index];
		p.pt.y = (float)kp[index + 1];
		keypoints[i] = p;
	}
}
void SuperPoint::compute(InputArray image,
	std::vector<KeyPoint>& keypoints,
	OutputArray descriptors)
{

	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "SuperPoint");
	Ort::SessionOptions sessionOptions;
	sessionOptions.SetIntraOpNumThreads(1);
	sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
	static Ort::Session extractorSession(env, this->m_modelPath.c_str(), sessionOptions);

	Mat img = image.getMat();
	Mat grayImg;
	cvtColor(img, grayImg, COLOR_BGR2GRAY);
	float mean, std;

	vector<float> imgData = ApplyTransform(grayImg, mean, std);

	vector<int64_t> inputShape{ 1, 1, grayImg.rows, grayImg.cols };

	Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, imgData.data(), imgData.size(), inputShape.data(), inputShape.size());

	const char* input_names[] = { "image" };
	const char* output_names[] = { "keypoints","scores","descriptors" };
	Ort::RunOptions run_options;
	vector<Ort::Value> outputs = extractorSession.Run(run_options, input_names, &inputTensor, 1, output_names, 3);

	std::vector<int64_t> kpshape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
	int64* kp = (int64*)outputs[0].GetTensorMutableData<void>();
	int keypntcounts = kpshape[1];
	keypoints.resize(keypntcounts);
	for (int i = 0; i < keypntcounts; i++)
	{
		KeyPoint p;
		int index = i * 2;
		p.pt.x = (float)kp[index];
		p.pt.y = (float)kp[index + 1];
		keypoints[i] = p;
	}

	std::vector<int64_t> desshape = outputs[2].GetTensorTypeAndShapeInfo().GetShape();
	float* des = (float*)outputs[2].GetTensorMutableData<void>();
	Mat desmat = descriptors.getMat();
	desmat.create(Size(desshape[2], desshape[1]), CV_32FC1);
	for (int h = 0; h < desshape[1]; h++)
	{
		for (int w = 0; w < desshape[2]; w++)
		{
			int index = h * desshape[2] + w;
			desmat.at<float>(h, w) = des[index];
		}
	}
	desmat.copyTo(descriptors);
}
