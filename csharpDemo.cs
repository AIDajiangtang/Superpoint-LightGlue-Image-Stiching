using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using MathNet.Numerics.Statistics;
using System.Windows.Media.Imaging;

namespace LightGlue
{
    /// <summary>
    /// MainWindow.xaml 的交互逻辑
    /// </summary>
    public partial class MainWindow : Window
    {     
        public MainWindow()
        {
            InitializeComponent();
            var extractor = new InferenceSession("D:\\superpoint.onnx");
            Transforms tranform = new Transforms(512);

            BitmapImage bitmap1 = new BitmapImage(new Uri("D:\\1.jpg"));
            float mean1 = 0.0f;
            float std1 = 0.0f;
            float[] img1 = tranform.ApplyImage("D:\\1.jpg", (int)bitmap1.Width, (int)bitmap1.Height,ref mean1,ref std1);
            // Define the input tensor
            var image1 = new DenseTensor<float>(img1, new[] { 1, 1, 512, 512 });
            // Create a dictionary to specify named inputs
            var inputs1 = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("image", image1)
            };

            float[] k1 = null;
            int[] kp1d =null;
            Tensor<float> kp1des = null;
            // Run the inference
            var results1 = extractor.Run(inputs1);
            // Get the output tensors
            var keypoints1 = results1.First(o => o.Name == "keypoints").AsTensor<long>();
            kp1d = keypoints1.Dimensions.ToArray();
            var lkp1 = keypoints1.ToList();
            k1 = new float[lkp1.Count];
            for (int i = 0; i < lkp1.Count; i++)
            {
                k1[i] = (lkp1[i] - 256) / 256.0f;
            }
            var scores1 = results1.First(o => o.Name == "scores").AsTensor<float>();
            kp1des = results1.First(o => o.Name == "descriptors").AsTensor<float>();


            BitmapImage bitmap2 = new BitmapImage(new Uri("D:\\2.jpg"));
            float mean2 = 0.0f;
            float std2 = 0.0f;
            float[] img2 = tranform.ApplyImage("D:\\2.jpg", (int)bitmap2.Width, (int)bitmap2.Height, ref mean2, ref std2);
            // Define the input tensor
            var image2 = new DenseTensor<float>(img2, new[] { 1, 1, 512, 512 });
            // Create a dictionary to specify named inputs
            var inputs2 = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("image", image2)
            };
            float[] k2 = null;
            int[] kp2d = null;
            Tensor<float> kp2des = null;
            // Run the inference
            var results2 = extractor.Run(inputs2);
            // Get the output tensors
            var keypoints2 = results2.First(o => o.Name == "keypoints").AsTensor<long>();
            kp2d = keypoints2.Dimensions.ToArray();
            var lkp2 = keypoints2.ToList();
            k2 = new float[lkp2.Count];
            for (int i = 0; i < lkp2.Count; i++)
            {
                k2[i] = (lkp2[i] - 256) / 256.0f;
            }
            var scores2 = results2.First(o => o.Name == "scores").AsTensor<float>();
            kp2des = results2.First(o => o.Name == "descriptors").AsTensor<float>();

            var lightglue = new InferenceSession("D:\\superpoint_lightglue.onnx");

            var kp1t = new DenseTensor<float>(k1, kp1d);
            var kp2t = new DenseTensor<float>(k2, kp2d);
            var inputs3 = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("kpts0", kp1t),
                NamedOnnxValue.CreateFromTensor("kpts1", kp2t),
                NamedOnnxValue.CreateFromTensor("desc0", kp1des),
                NamedOnnxValue.CreateFromTensor("desc1", kp2des)
            };

            using (var results = lightglue.Run(inputs3))
            {
                var match0 = results.First(o => o.Name == "matches0").AsTensor<Int64>();

                List<Int64> mt1 = new List<Int64>();
                List<Int64> mt2 = new List<Int64>();
                var matchList = match0.ToList();
                for (int i =0;i< matchList.Count;i++)
                {
                    if (matchList[i] > -1)
                    {
                        mt1.Add(i);
                        mt2.Add(matchList[i]);
                    }
                       
                }


                //// Create a new bitmap with the desired size
                System.Drawing.Bitmap bitmapout = new System.Drawing.Bitmap(1024, 512);
                //// Set the pixel values in the bitmap
                for (int y = 0; y < 512; y++)
                {
                    for (int x = 0; x < 512; x++)
                    {
                        int ind = y * 512 + x;
                        int value1 = (int)(img1[ind] * std1 + mean1);
                        if (value1 < 0) value1 = 0;
                        if (value1 > 255) value1 = 255;
                        bitmapout.SetPixel(x, y, System.Drawing.Color.FromArgb(value1, value1, value1));

                        int value2 = (int)(img2[ind] * std2 + mean2);
                        if (value2 < 0) value2 = 0;
                        if (value2 > 255) value2 = 255;
                        bitmapout.SetPixel(x+512, y, System.Drawing.Color.FromArgb(value2, value2, value2));
                    }
                }

                for (int i =0;i< mt2.Count;i++)
                {
                    int inex1 = (int)mt1[i];                 
                    bitmapout.SetPixel((int)lkp1[2 * inex1], (int)lkp1[2 * inex1+1], System.Drawing.Color.FromArgb(255,0,0));

                    int inex2 = (int)mt2[i];
                    bitmapout.SetPixel((int)lkp2[2 * inex2]+512, (int)lkp2[2 * inex2 + 1], System.Drawing.Color.FromArgb(255, 0, 0));

                }

                bitmapout.Save("D:\\lightglue.png");

            }

        }
    }


    class Transforms
    {
        public Transforms(int target_length)
        {
            this.mTargetLength = target_length;
        }
        /// <summary>
        /// 变换图像，将原始图像变换大小
        /// </summary>
        /// <returns></returns>
        public float[] ApplyImage(string filename, int orgw, int orgh,ref float mean,ref float std)
        {
            int neww = 0;
            int newh = 0;
            this.GetPreprocessShape(orgw, orgh, this.mTargetLength, ref neww, ref newh);

            float[,,] resizeImg = this.Resize(filename, neww, newh);

            //计算均值
            float[] means = new float[resizeImg.GetLength(0)];
            for (int i = 0; i < resizeImg.GetLength(0); i++)
            {
                float[] data = new float[resizeImg.GetLength(1) * resizeImg.GetLength(2)];
                for (int j = 0; j < resizeImg.GetLength(1); j++)
                {
                    for (int k = 0; k < resizeImg.GetLength(2); k++)
                    {
                        data[j * resizeImg.GetLength(2) + k] = resizeImg[i, j, k];
                    }
                }
                means[i] = (float)MathNet.Numerics.Statistics.Statistics.Mean(data);
            }

            //计算标准差
            float[] stdDev = new float[resizeImg.GetLength(0)];
            for (int i = 0; i < resizeImg.GetLength(0); i++)
            {
                float[] data = new float[resizeImg.GetLength(1) * resizeImg.GetLength(2)];
                for (int j = 0; j < resizeImg.GetLength(1); j++)
                {
                    for (int k = 0; k < resizeImg.GetLength(2); k++)
                    {
                        data[j * resizeImg.GetLength(2) + k] = resizeImg[i, j, k];
                    }
                }
                stdDev[i] = (float)MathNet.Numerics.Statistics.Statistics.StandardDeviation(data);
            }


            float[] transformedImg = new float[this.mTargetLength * this.mTargetLength];
            for (int i = 0; i < neww; i++)
            {
                for (int j = 0; j < newh; j++)
                {
                    int index = j * this.mTargetLength + i;
                    transformedImg[index] = (resizeImg[0, i, j] - means[0]) / stdDev[0];
                }
            }
            mean = means[0];
            std = stdDev[0];
            return transformedImg;
        }
        float[,,] Resize(string filename, int neww, int newh)
        {

            //加载原始图像
            Image originalImage = Image.FromFile(filename);

            //创建新的Bitmap对象，并设置大小
            Bitmap resizedImage = new Bitmap(neww, newh);

            //创建Graphics对象，并设置插值模式
            Graphics g = Graphics.FromImage(resizedImage);
            g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;

            //将原始图像绘制到新的Bitmap对象中
            g.DrawImage(originalImage, new Rectangle(0, 0, neww, newh), new Rectangle(0, 0, originalImage.Width, originalImage.Height), GraphicsUnit.Pixel);

            //保存新的图像
            //resizedImage.Save("resized.jpg", System.Drawing.Imaging.ImageFormat.Jpeg);
            float[,,] newimg = new float[3, neww, newh];
            for (int i = 0; i < neww; i++)
            {
                for (int j = 0; j < newh; j++)
                {
                    newimg[0, i, j] = resizedImage.GetPixel(i, j).R;
                    newimg[1, i, j] = resizedImage.GetPixel(i, j).G;
                    newimg[2, i, j] = resizedImage.GetPixel(i, j).B;
                }
            }
            //释放资源
            originalImage.Dispose();
            resizedImage.Dispose();
            g.Dispose();

            return newimg;
        }

      

        void GetPreprocessShape(int oldw, int oldh, int long_side_length, ref int neww, ref int newh)
        {
            float scale = long_side_length * 1.0f / Math.Max(oldh, oldw);
            float newht = oldh * scale;
            float newwt = oldw * scale;

            neww = (int)(newwt + 0.5);
            newh = (int)(newht + 0.5);
        }

        int mTargetLength;//目标图像大小（宽=高）


    }
}
