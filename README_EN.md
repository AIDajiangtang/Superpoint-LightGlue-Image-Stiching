English | [简体中文](README.md)
# DeepLearning-based-Feature-extraction-and-matching
Integrate SuperPoint and LightGlue into OpenCV image stitching algorithm<br />  

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

Now, our protagonist comes on stage. We need to add a deep learning feature extraction algorithm, SuperPoint, and a deep learning feature matching algorithm, LightGlue, to OpenCV.<br />  

SuperPoint:<br />  

Paper: https://arxiv.org/pdf/1712.07629.pdf<br />  

Official code: https://github.com/rpautrat/SuperPoint<br />  

LightGlue:<br />  

Paper: https://arxiv.org/pdf/2306.13643.pdf<br />  

Official code: https://github.com/cvg/LightGlue<br />  

According to the inheritance system of OpenCV classes, the base class of the feature extractor is Feature2D, and the base class of the feature matcher is FeaturesMatcher. We add two classes based on this: SuperPoint and LightGlue, and re-implement the virtual methods of the base class.<br />  

[superpoint](superpoint.cpp)<br />  
[lightglue](lightglue.cpp)<br />  

After that, set the newly added classes to the stitching pipeline.<br />  
[CPPDemo](cppDemo.cpp)<br />  

For those who are not familiar with C++, I also provide a C#Demo.<br />  
[CSharpDemo](csharpDemo.cs)<br />  

Follow our WeChat official account: 人工智能大讲堂<br />  
<img width="180" src="https://user-images.githubusercontent.com/18625471/228743333-77abe467-2385-476d-86a2-e232c6482291.jpg"><br /> 
Reply "sl" in the background to obtain the pre-trained models mentioned above.
