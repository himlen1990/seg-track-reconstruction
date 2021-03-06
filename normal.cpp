//#include "stdafx.h"
#include<opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;
int main(int argv, char** argc)
{
  Mat depth=imread("depth.png");
  imshow("depth",depth);

  if(depth.type() != CV_32FC1)
    depth.convertTo(depth, CV_32FC1);

  Mat normals(depth.size(), CV_32FC3);

  for(int x = 1; x < depth.cols - 1; ++x)
    {
      for(int y = 1; y < depth.rows - 1; ++y)
	{
	  // 3d pixels, think (x,y, depth)
	  /* * * * *
	   * * t * *
	   * l c * *
	   * * * * */

	  Vec3f t(x,y-1,depth.at<float>(y-1, x));
	  Vec3f l(x-1,y,depth.at<float>(y, x-1));
	  Vec3f c(x,y,depth.at<float>(y, x));

	  Vec3f d = (l-c).cross(t-c);

	  Vec3f n = normalize(d);
	  normals.at<Vec3f>(y,x) = n;

	}
    }

  cout<<normals<<endl;
  imshow("explicitly cross_product normals", normals);


  waitKey(0);
}
