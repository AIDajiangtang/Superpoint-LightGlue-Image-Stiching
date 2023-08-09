简体中文 | [English](README_EN.md)

# DeepLearning-based-Feature-extraction-and-matching
将深度学习预训练模型 SuperPoint 和 LightGlue 集成到OpenCV拼接算法中。

 ## OpenCV拼接流水线</h2>
图像特征提取与匹配是许多高级计算机视觉任务的基础，例如图像配准，图像拼接，相机矫正，SLAM，深度估计等等。
今天我们就以图像拼接为切入点，来看一下特征提取与匹配的重要性。

OpenCV中提供了封装程度非常高的Stitcher类，通过下面几行代码就能实现图像拼接。  
Mat pano;  
Ptr<Stitcher> stitcher = Stitcher::create(mode);  
Stitcher::Status status = stitcher->stitch(imgs, pano);  

但图像拼接整个过程非常复杂。  
用文字总结一下拼接算法的主要流程：  
特征提取->特征匹配->评估相机参数->生成融合图像  

其中特征提取最为重要，特征点和特征描述符的质量好坏决定了最终的拼接效果。 

目前OpenCV中提供了SIFT，SURF，ORB等特征提取器。  
目前OpenCV中提供了Brute-Force，FLANN，KNN等特征匹配器。  

然后，通过下面代码将选择的特征提取和匹配算法设置到拼接流水线中。  

if (feature_type == FeatureType::SURF)  
    stitcher->setFeaturesFinder(xfeatures2d::SURF::create());  
  else if (feature_type == FeatureType::SIFT)  
    stitcher->setFeaturesFinder(SIFT::create());  
  else   
    stitcher->setFeaturesFinder(ORB::create());  
    
stitcher->setFeaturesMatcher(makePtr<detail::BestOf2NearestMatcher>( false));  
stitcher->setFeaturesMatcher(makePtr<detail::BestOf2NearestRangeMatcher>( false));  
stitcher->setFeaturesMatcher(makePtr<detail::AffineBestOf2NearestMatcher>(true, false));  

这要感谢C++面向对象编程的思想，通过继承与多态来实现不同特征提取和匹配算法的扩展。  

 ## 深度学习特征检测匹配算法</h2>  
说到这里该我们的主角出场了，我们要为OpenCV追加一种深度学习特征提取算法：SuperPoint，以及深度学习特征匹配算法：LightGlue。  
SuperPoint：​  
[[`Paper`](https://arxiv.org/pdf/1712.07629.pdf)] [[`源码`](https://github.com/rpautrat/SuperPoint )]  

​
LightGlue:  
[[`Paper`](https://arxiv.org/pdf/2306.13643.pdf )] [[`源码`](https://github.com/cvg/LightGlue)]  


根据OpenCV中类继承体系，特征提取类的基类为Feature2D，特征匹配的基类为FeaturesMatcher，我们以此为基类新增两个类：SuperPoint和LightGlue，并重新实现基类的虚方法。  
[superpoint](superpoint.cpp)  
[lightglue](lightglue.cpp)  

 ## 如何编译</h2>  
目前仅在Windows 11，Visualstudio2019，Cmake3.26.4中测试成功   
1.微信公众号回复【sl】获取预训练模型 ，模型路径通过参数传递给程序  
微信公众号：**人工智能大讲堂**  
<img width="180" src="https://user-images.githubusercontent.com/18625471/228743333-77abe467-2385-476d-86a2-e232c6482291.jpg">  

后台回复【sl】获取上面的预训练模型和第三方依赖库。  
2.微信公众号回复【sl】获取第三方库：OpenCV和ONNXRuntime，然后将其解压到源码目录，OPenCV是我用Visual Studio2019编译的。ONNXRuntime不需要自己编译，下载官网编译好的即可  
最终的项目结构如下：  
project_root/  
  |- common.h  
  |- superpoint.h  
  |- superpoint.cpp  
  |- lightglue.h  
  |- lightglue.cpp  
  |- cppDemo.cpp  
  |- opencv/  
  |- onnxruntime-win-x64-1.15.1/  
  |- CMakeLists.txt  

3.打开Cmake，输入源码路径和编译输出路径，然后以此点积Config->Generate->Open Project  

 ## 集成到OpenCV中</h2>  
然后将新增加的类设置到拼接流水线中。  
[CPPDemo](cppDemo.cpp)  
 std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;  
 std::wstring sp = converter.from_bytes(superPointPath);  
 std::wstring lh = converter.from_bytes(lightGluePath);  
 
 Mat pano;  
 Ptr<Stitcher> stitcher = Stitcher::create(mode);      
 Ptr<SuperPoint> superpointp = makePtr&lt;SuperPoint&gt;(sp);  
 Ptr<LightGlue> lightglue = makePtr&lt;LightGlue&gt;(lh, mode);  
 stitcher->setPanoConfidenceThresh(0.1f);   
 stitcher->setFeaturesFinder(superpointp);//SpuerPoint feature extraction  
 stitcher->setFeaturesMatcher(lightglue);//LightGlue feature matching   
 Stitcher::Status status = stitcher->stitch(imgs, pano); 

1.将onnxruntime.dll和opencv_world455.dll（opencv_world455d.dll debug模式）放到exe路径下  
2.运行cppDemo.exe，并设置参数，例如--mode panorama --lg D:\\superpoint_lightglue.onnx --sp D:\\superpoint.onnx D:\\1.jpg D:\2.jpg  

<img width="500" src="https://user-images.githubusercontent.com/18625471/256421932-94e8b07b-fc4b-4307-a94e-e7e735d620d8.jpg">  

拼接支持两种变换模型，仿射变换和透视变换，由--mode (panorama|scans)指定，panorama表示透视变换模型，scans代表仿射变换模型  
--sp 指定superpoint onnx格式模型路径  
--lg 指定lightflue onnx格式模型路径  
 D:\\1.jpg D:\2.jpg为拼接输入图像  

拼接结果  
 <img width="500" src="https://user-images.githubusercontent.com/18625471/256420139-3c03fbcb-3047-44a5-9403-d98f86e222da.jpg">  
特征匹配  
<img width="800" src="https://user-images.githubusercontent.com/18625471/256420458-c296cd92-ddbc-479d-a224-ba01a56450f5.jpg">  
 
对于不熟悉C++的，我还提供了C#Demo  
[CSharpDemo](csharpDemo.cs)  

