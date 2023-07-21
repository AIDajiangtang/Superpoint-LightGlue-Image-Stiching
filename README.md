简体中文 | [English](README_EN.md)

# DeepLearning-based-Feature-extraction-and-matching
将深度学习预训练模型 SuperPoint 和 LightGlue 集成到OpenCV拼接算法中。<br />  

 ## OpenCV拼接流水线</h2>
图像特征提取与匹配是许多高级计算机视觉任务的基础，例如图像配准，图像拼接，相机矫正，SLAM，深度估计等等。<br />  
今天我们就以图像拼接为切入点，来看一下特征提取与匹配的重要性。<br />  

OpenCV中提供了封装程度非常高的Stitcher类，通过下面几行代码就能实现图像拼接。<br />  
Mat pano;<br />  
Ptr<Stitcher> stitcher = Stitcher::create(mode);<br />  
Stitcher::Status status = stitcher->stitch(imgs, pano);<br />  

但图像拼接整个过程非常复杂。<br />  
用文字总结一下拼接算法的主要流程：<br />  
特征提取->特征匹配->评估相机参数->生成融合图像<br />  

其中特征提取最为重要，特征点和特征描述符的质量好坏决定了最终的拼接效果。<br />  

目前OpenCV中提供了SIFT，SURF，ORB等特征提取器。<br />  
目前OpenCV中提供了Brute-Force，FLANN，KNN等特征匹配器。<br />  

然后，通过下面代码将选择的特征提取和匹配算法设置到拼接流水线中。<br />  

if (feature_type == FeatureType::SURF)<br />  
    stitcher->setFeaturesFinder(xfeatures2d::SURF::create());<br />  
  else if (feature_type == FeatureType::SIFT)<br />  
    stitcher->setFeaturesFinder(SIFT::create());<br />  
  else<br />  
    stitcher->setFeaturesFinder(ORB::create());<br />  
    
stitcher->setFeaturesMatcher(makePtr<detail::BestOf2NearestMatcher>( false));<br />  
stitcher->setFeaturesMatcher(makePtr<detail::BestOf2NearestRangeMatcher>( false));<br />  
stitcher->setFeaturesMatcher(makePtr<detail::AffineBestOf2NearestMatcher>(true, false));<br />  

这要感谢C++面向对象编程的思想，通过继承与多态来实现不同特征提取和匹配算法的扩展。<br />  

 ## 深度学习特征检测匹配算法</h2><br />  
说到这里该我们的主角出场了，我们要为OpenCV追加一种深度学习特征提取算法：SuperPoint，以及深度学习特征匹配算法：LightGlue。<br />  
SuperPoint：<br />  ​
论文地址：https://arxiv.org/pdf/1712.07629.pdf<br />  
官方源码：https://github.com/rpautrat/SuperPoint<br />  
​
LightGlue:<br />  
论文地址：https://arxiv.org/pdf/2306.13643.pdf<br />  
官方源码：https://github.com/cvg/LightGlue<br />  

根据OpenCV中类继承体系，特征提取类的基类为Feature2D，特征匹配的基类为FeaturesMatcher，我们以此为基类新增两个类：SuperPoint和LightGlue，并重新实现基类的虚方法。<br />  
[superpoint](superpoint.cpp)<br />  
[lightglue](lightglue.cpp)<br />  

 ##如何编译</h2>
目前仅在Windows 11，Visualstudio2019，Cmake3.26.4中测试成功 <br />  
1.微信公众号回复【sl】获取预训练模型，将模型放到D盘根目录（或者将dll以资源的方式存储到dll中）
微信公众号：**人工智能大讲堂**<br />  
<img width="180" src="https://user-images.githubusercontent.com/18625471/228743333-77abe467-2385-476d-86a2-e232c6482291.jpg"><br /> 
后台回复【sl】获取上面的预训练模型和第三方依赖库。<br />  
2.微信公众号回复【sl】获取第三方库：OpenCV和ONNXRuntime，然后将其解压到源码目录，OPenCV是我用Visual Studio2019编译的。ONNXRuntime不需要自己编译，下载官网编译好的即可<br />  
最终的项目结构如下：<br /> 

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

3.打开Cmake，输入源码路径和编译输出路径，然后以此点积Config，Generate，Open Project

 ## 集成到OpenCV中</h2><br />  
然后将新增加的类设置到拼接流水线中。<br />  
[CPPDemo](cppDemo.cpp)<br />  
Mat pano;<br />  
Ptr<Stitcher> stitcher = Stitcher::create(mode);<br />  
stitcher->setFeaturesFinder(makePtr&lt;SuperPoint&gt;());//SpuerPoint feature extraction<br />  
stitcher->setFeaturesMatcher(makePtr&lt;LightGlue&gt;());//LightGlue feature matching<br />  
Stitcher::Status status = stitcher->stitch(imgs, pano);<br />  

对于不熟悉C++的，我还提供了C#Demo<br />  
[CSharpDemo](csharpDemo.cs)<br />  

