English | [简体中文](README.md)
# DeepLearning-based-Feature-extraction-and-matching
Integrate SuperPoint and LightGlue into OpenCV image stitching algorithm<br />  

 ## OpenCV stitching pipeline</h2>
Image feature extraction and matching are the foundation of many advanced computer vision tasks, such as image registration, image stitching, camera calibration, SLAM, depth estimation, etc.<br />  

Today, let's take image stitching as an entry point to see the importance of feature extraction and matching.<br />  

OpenCV provides the highly encapsulated Stitcher class, which can achieve image stitching with just a few lines of code:<br />  

Mat pano;<br />  
Ptr stitcher = Stitcher::create(mode);<br />  
Stitcher::Status status = stitcher->stitch(imgs, pano);<br />  
However, the entire process of image stitching is very complex.<br />  

Summarizing the main process of the stitching algorithm in words:<br />  

Feature extraction → feature matching → evaluation of camera parameters → generation of fused image<br />  

Among them, feature extraction is the most important, and the quality of feature points and feature descriptors determines the final stitching effect.<br />  

Currently, OpenCV provides feature extractors such as SIFT, SURF, and ORB.<br />  

Currently, OpenCV provides feature matchers such as Brute-Force, FLANN, and KNN.<br />  

Then, set the selected feature extraction and matching algorithms to the stitching pipeline using the following code:<br />  

if (feature_type == FeatureType::SURF)<br />  
    stitcher->setFeaturesFinder(xfeatures2d::SURF::create());<br />  
else if (feature_type == FeatureType::SIFT)<br />  
    stitcher->setFeaturesFinder(SIFT::create());<br />  
else<br />  
    stitcher->setFeaturesFinder(ORB::create());<br />  

stitcher->setFeaturesMatcher(makePtr<detail::BestOf2NearestMatcher>(false));<br />  
stitcher->setFeaturesMatcher(makePtr<detail::BestOf2NearestRangeMatcher>(false));<br />  
stitcher->setFeaturesMatcher(makePtr<detail::AffineBestOf2NearestMatcher>(true, false));<br />  
Thanks to the object-oriented programming concept of C++, different feature extraction and matching algorithms can be implemented through inheritance and polymorphism.<br />  

 ## Deep Learning Feature Detection and Matching Algorithm</h2>  
Now, our protagonist comes on stage. We need to add a deep learning feature extraction algorithm, SuperPoint, and a deep learning feature matching algorithm, LightGlue, to OpenCV.<br />  

SuperPoint：​  
[[`Paper`](https://arxiv.org/pdf/1712.07629.pdf)] [[`Source Code`](https://github.com/rpautrat/SuperPoint )]  

​
LightGlue:  
[[`Paper`](https://arxiv.org/pdf/2306.13643.pdf )] [[`Source Code`](https://github.com/cvg/LightGlue)]  

According to the inheritance system of OpenCV classes, the base class of the feature extractor is Feature2D, and the base class of the feature matcher is FeaturesMatcher. We add two classes based on this: SuperPoint and LightGlue, and re-implement the virtual methods of the base class.<br />  

[superpoint](superpoint.cpp)<br />  
[lightglue](lightglue.cpp)<br />  

 ## how to compile</h2>  
Currently only tested successfully in Windows 11, Visualstudio2019, Cmake3.26.4 <br />
1. Reply [sl] from the WeChat official account to obtain the pre-trained model, and put the model in the root directory of the D drive (or store the dll in the dll as a resource)
2. WeChat public account reply [sl] Obtain third-party libraries: OpenCV and ONNXRuntime, and then decompress them to the source code directory. I compiled OPenCV with Visual Studio2019. ONNXRuntime does not need to be compiled by itself, just download the compiled one from the official website<br />
The final project structure is as follows:<br />
project_root/
   |- common.h
   |- superpoint.h
   |- superpoint.cpp
   |- lightglue.h
   |- lightglue.cpp
   |- cppDemo.cpp
   |- opencv/
   |-onnxruntime-win-x64-1.15.1/
   |- CMakeLists.txt

3. Open Cmake, enter the source code path and the compiled output path, and then use this dot product Config, Generate, Open Project

 ## Integrate into OpenCV</h2>    
[CPPDemo](cppDemo.cpp)<br />  
Mat pano;<br />  
Ptr<Stitcher> stitcher = Stitcher::create(mode);<br />  
stitcher->setFeaturesFinder(makePtr&lt;SuperPoint&gt;());//SpuerPoint feature extraction<br />  
stitcher->setFeaturesMatcher(makePtr&lt;LightGlue&gt;());//LightGlue feature matching<br />  
Stitcher::Status status = stitcher->stitch(imgs, pano);<br />  
1.Put onnxruntime.dll and opencv_world455.dll (opencv_world455d.dll debug mode) in the exe path  
2.Run cppDemo.exe and set parameters, for example --mode panorama --lg D:\\superpoint_lightglue.onnx --sp D:\\superpoint.onnx D:\\1.jpg D:\2.jpg  
<img width="500" src="https://user-images.githubusercontent.com/18625471/256421932-94e8b07b-fc4b-4307-a94e-e7e735d620d8.jpg">  

Stitching supports two transformation models, affine transformation and perspective transformation, specified by --mode (panorama|scans), panorama represents the perspective transformation model, and scans represents the affine transformation model

--sp specifies the superpoint onnx format model path  
--lg specifies the lightflue onnx format model path  
D:\1.jpg D:\2.jpg is input image for splicing  

stiching result  
 <img width="500" src="https://user-images.githubusercontent.com/18625471/256420139-3c03fbcb-3047-44a5-9403-d98f86e222da.jpg">  
feature matching  
<img width="800" src="https://user-images.githubusercontent.com/18625471/256420458-c296cd92-ddbc-479d-a224-ba01a56450f5.jpg">  

For those who are not familiar with C++, I also provide a C#Demo.<br />  
[CSharpDemo](csharpDemo.cs)<br />  

Follow our WeChat official account: 人工智能大讲堂<br />  
<img width="180" src="https://user-images.githubusercontent.com/18625471/228743333-77abe467-2385-476d-86a2-e232c6482291.jpg"><br /> 
Reply "sl" in the background to obtain the pre-trained models and third-party dependent libraries mentioned above.<br /> 
