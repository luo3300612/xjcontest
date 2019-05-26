#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

const int patchSize = 5;
const double w = 0.95;

namespace fs = std::filesystem;

void guidedFilter(Mat &source, Mat &guided_image, Mat &output, int radius,
                  float epsilon) {
  CV_Assert(radius >= 2 && epsilon > 0);
  CV_Assert(source.data != NULL && source.channels() == 1);
  CV_Assert(guided_image.channels() == 1);
  CV_Assert(source.rows == guided_image.rows &&
            source.cols == guided_image.cols);

  Mat guided;
  if (guided_image.data == source.data) {
    // make a copy
    guided_image.copyTo(guided);
  } else {
    guided = guided_image;
  }

  //将输入扩展为32位浮点型，以便以后做乘法

  //计算I*p和I*I
  Mat mat_Ip, mat_I2;
  multiply(guided_image, source, mat_Ip);
  multiply(guided_image, guided_image, mat_I2);

  //计算各种均值
  Mat mean_p, mean_I, mean_Ip, mean_I2;
  Size win_size(2 * radius + 1, 2 * radius + 1);
  boxFilter(source, mean_p, CV_32F, win_size);
  boxFilter(guided_image, mean_I, CV_32F, win_size);
  boxFilter(mat_Ip, mean_Ip, CV_32F, win_size);
  boxFilter(mat_I2, mean_I2, CV_32F, win_size);

  //计算Ip的协方差和I的方差
  Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);
  Mat var_I = mean_I2 - mean_I.mul(mean_I);
  var_I += epsilon;

  //求a和b
  Mat a, b;
  divide(cov_Ip, var_I, a);
  b = mean_p - a.mul(mean_I);

  //对包含像素i的所有a、b做平均
  Mat mean_a, mean_b;
  boxFilter(a, mean_a, CV_32F, win_size);
  boxFilter(b, mean_b, CV_32F, win_size);

  //计算输出 (depth == CV_32F)
  output = mean_a.mul(guided_image) + mean_b;
}

void dehaze(Mat &img, Mat_<Vec3f> &J) {
  Mat aChannels[3];
  int cols = img.cols;
  int rows = img.rows;
  // resize(img,img,{600,400});
  cv::split(img, aChannels);
  // imshow("WindowB",aChannels[0]);
  // imshow("WindowG", aChannels[1]);
  // imshow("WindowR", aChannels[2]);
  // cout << img.cols << endl;
  // cout << img.rows << endl;
  Mat_<uchar> darkChannel;
  darkChannel = min(aChannels[0], aChannels[1]);
  darkChannel = min(darkChannel, aChannels[2]);
  // cout << (int)darkChannel.at<uchar>(1, 0) << endl;
  darkChannel = darkChannel.reshape(1, 1);

  // cout << (int)darkChannel.at<uchar>(0, 450) << endl;
  // cout << darkChannel.rows << endl;
  // cout << darkChannel.cols << endl;
  // TODO 表示一列
  Mat_<int> index1;
  sortIdx(darkChannel, index1, SORT_DESCENDING);
  // cout << index << endl;
  //cout << (int)darkChannel(index1(0)) << endl;
  //cout << (int)darkChannel(index1(1)) << endl;

  int split = (int)(darkChannel.cols * 0.001);
  //cout << split << endl;

  Mat_<uchar> tmp(1, split);
  Vec3d A;
  for (int c = 0; c < 3; c++) {
    for (int i = 0; i < split; i++) {
      tmp(i) = aChannels[c].reshape(1, 1).at<uchar>(index1(i));
    }
    Mat_<int> index2;
    sortIdx(tmp, index2, SORT_DESCENDING);
    A(c) = (double)aChannels[c].reshape(1, 1).at<uchar>((int)index1(index2(0)));
  }

  
  //cout << A << endl;

  // cout << (int)darkChannel((int)index1(index2(0))) << endl;
  // imshow("Window", darkChannel);
  // waitKey(0);
  Mat_<_Float32> t(rows, cols);
  for (int i = 0; i < cols / patchSize; i++) {
    for (int j = 0; j < rows / patchSize; j++) {
      Mat patch = img({i * patchSize, j * patchSize, patchSize, patchSize}).clone();
      Mat patchChannels[3];
      cv::split(patch, patchChannels);
      Mat_<double> bChannel;
      Mat_<double> grChannel;
      Mat_<double> rChannel;
      bChannel = patchChannels[0] / A(0);
      grChannel = patchChannels[1] / A(1);
      rChannel = patchChannels[2] / A(2);
      double *bMinVal = new double[1];
      double *grMinVal = new double[1];
      double *rMinVal = new double[1];
      minMaxLoc(bChannel, bMinVal);
      minMaxLoc(grChannel, grMinVal);
      minMaxLoc(rChannel, rMinVal);
      double minVal = min(min(*bMinVal, *grMinVal), *rMinVal);
      t({i * patchSize, j * patchSize, patchSize, patchSize}) =
          (1 - w * minVal) * 255;

      delete bMinVal;
      delete grMinVal;
      delete rMinVal;
    }
  }
  
  // imshow("Window", t);
  // imshow("Window1", img);
  // waitKey(0);
  Mat_<uchar> imgGray;
  cvtColor(img, imgGray, COLOR_BGR2GRAY);
  Mat_<_Float32> imgGrayF32 = imgGray;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      imgGrayF32(i, j) = imgGray(i, j) / 255.;
    }
  }
  Mat_<_Float32> t_refine;
  guidedFilter(t, imgGrayF32, t_refine, 6 * patchSize, 0.001);

  Mat_<uchar> t_u8;
  Mat_<uchar> t_refine_u8;

  t.convertTo(t_u8, CV_8U);
  t_refine.convertTo(t_refine_u8, CV_8U);

  // imshow("Window", t_u8);
  // imshow("Window1", t_refine_u8);

  float t0 = 0.1;

  Mat_<Vec3f> imgF32;
  img.convertTo(imgF32, CV_32FC3);
  Vec3f AF32 = A / 255.;
  //cout << AF32 << endl;

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      J(i, j)
      (0) = (imgF32(i, j)(0) - AF32(0)) / (max(t_refine(i, j), t0)) + AF32(0);
      J(i, j)
      (1) = (imgF32(i, j)(1) - AF32(1)) / (max(t_refine(i, j), t0)) + AF32(1);
      J(i, j)
      (2) = (imgF32(i, j)(2) - AF32(2)) / (max(t_refine(i, j), t0)) + AF32(2);
    }
  }
  // imshow("Window2", img);
  Mat before;
  normalize(J, J, 1, 0, NORM_MINMAX);

  // imshow("Window3", before);
  // imshow("Window3", J);
  float alpha = 1;
  float beta = 0;

  // Mat_<Vec3b> Ju8;
  // normalize(J, J, 1, 0, NORM_MINMAX);
  // J.convertTo(Ju8,CV_8UC3);

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      J(i, j)(0) = alpha * J(i, j)(0) + beta;
      J(i, j)(1) = alpha * J(i, j)(1) + beta;
      J(i, j)(2) = alpha * J(i, j)(2) + beta;
    }
  }
  // cout << Ju8 << endl;
  // // cout << J << endl;
  // aChannels[0].release();
  // aChannels[1].release();
  // aChannels[2].release();
  // darkChannel.release();
  // tmp.release();
  // t.release();
  // imgGray.release();
  // imgGrayF32.release();
  // t_refine.release();
  // t_u8.release();
  // t_refine_u8.release();
  // imgF32.release();
  // before.release();
}

void equalizeHist3(Mat &src, Mat &dst) {
  Mat src_u8;
  // for (int i = 0; i < src.rows; i++) {
  //   for (int j = 0; j < src.cols; j++) {
  //     src_u8(i, j)(0) = 255 * J(i, j)(0);
  //     src_u8(i, j)(1) = 255 * J(i, j)(1);
  //     src_u8(i, j)(2) = 255 * J(i, j)(2);
  //   }
  // }
  //
  vector<Mat> aChannels;
  src.convertTo(src_u8, CV_8UC3, 255);
  cv::split(src_u8, aChannels);

  // imshow("Window6", src_u8);
  equalizeHist(aChannels[0], aChannels[0]);
  equalizeHist(aChannels[1], aChannels[1]);
  equalizeHist(aChannels[2], aChannels[2]);
  cv::merge(aChannels, dst);
}

int main(int argc, char const *argv[]) {

  string path = "./imgs";
  string save_path = "./imgs_after";
  int count = 0;
  for (const auto &entry : fs::directory_iterator(path)) {
    //std::cout << entry.path() << std::endl;
    count += 1;
    Mat img = imread(entry.path());
    // Mat img = imread("./haze.png");

    imshow("Window41", img);
    Mat_<Vec3f> J(img.rows, img.cols);
    // //GaussianBlur(img, img, {3, 3}, 3);
    dehaze(img, J);
    imshow("Window42", J);
    Mat out;
    equalizeHist3(J, out);
    imshow("Window43", out);
    waitKey(0);
    string save_file = save_path + '/' + (string)entry.path().filename();
    // imwrite(save_file,out);
    
    cout << count << "/" << 40000 << endl;
    // imwrite();
    img.release();
    out.release();
    J.release();
  }
  cout << "Done" << endl;

  return 0;
}

/*	g++ dehaze.cpp -lopencv_core -lopencv_highgui
-lopencv_imgcodecs -lopencv_videoio -lopencv_video -lopencv_videostab
-lopencv_imgproc
*/

/*
g++ dehaze.cpp -lopencv_core -lopencv_highgui -lopencv_imgcodecs
-lopencv_videoio -lopencv_video -lopencv_videostab -lopencv_imgproc -std=c++17
-lstdc++fs
*/