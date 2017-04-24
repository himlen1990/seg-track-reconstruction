//#include "stdafx.h"
#include<opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/flann.hpp>
#include<ctime>
#include "gco/GCoptimization.h"
using namespace cv;
using namespace std;

class rgbdSegment
{

public:
  Mat _image;
  int _region_size;
  int _MaxboundaryMargin;
  int _imgHeight;
  int _imgWidth;
  Mat _imageLab;
  Mat _labels;
  Mat _maskfordebug;
  std::vector<cv::Vec3f> _background, _foreground;



  rgbdSegment(){
  }

  void  init(const cv::Mat &img, int region_size=20, int MaxboundaryMargin = 15 )//20,15
  {
    _image=img;
    _region_size = region_size;
    _MaxboundaryMargin = MaxboundaryMargin;
    _imgHeight = _image.rows;
    _imgWidth = _image.cols;

    cvtColor(_image,_imageLab,CV_RGB2Lab);

    getPotentialBG(_background);
  }

  float rgbDis(cv::Vec3b &v1,cv::Vec3b &v2)
  {
    return  cv::norm(v1,v2,cv::NORM_L2);
  }


  float calImgVar(const cv::Mat &img)
  {
    Mat imageGray;
    cv::cvtColor(img,imageGray,CV_RGB2GRAY);
    Scalar mean,dev; 
    cv::meanStdDev(imageGray, mean, dev);  
    float imgvar = dev.val[0]*dev.val[0];
    return imgvar;
  }

  void getPotentialBG(std::vector<cv::Vec3f> &background)
  {
    /* for(int i=0; i<_image.rows; i++)
      {
	background.push_back(cv::Vec3f(_imageLab.at<cv::Vec3b>(i,0)[0],
				       _imageLab.at<cv::Vec3b>(i,0)[1],
				       _imageLab.at<cv::Vec3b>(i,0)[2]));
	background.push_back(cv::Vec3f(_imageLab.at<cv::Vec3b>(i,_imgWidth-1)[0],
				       _imageLab.at<cv::Vec3b>(i,_imgWidth-1)[1],
				       _imageLab.at<cv::Vec3b>(i,_imgWidth-1)[2]));
				       }*/
	
    /*    for(int i=0; i<_image.cols; i++)
      {
	background.push_back(cv::Vec3f(_imageLab.at<cv::Vec3b>(0,i)[0],
				       _imageLab.at<cv::Vec3b>(0,i)[1],
				       _imageLab.at<cv::Vec3b>(0,i)[2]));
	background.push_back(cv::Vec3f(_imageLab.at<cv::Vec3b>(_imgHeight-1,i)[0],
				       _imageLab.at<cv::Vec3b>(_imgHeight-1,i)[1],
				       _imageLab.at<cv::Vec3b>(_imgHeight-1,i)[2]));
				       }*/

  }
  
  void process()
  {

    int numPixels = _imgHeight*_imgWidth;

    int numLabels = 2;
    int *result = new int[numPixels];
    float *data = new float[numPixels*numLabels];
    float rgbdisBG;
    float rgbdisFG;


    flann::KDTreeIndexParams indexParams(2);
    flann::Index kdtreeBG(cv::Mat(_background).reshape(1), indexParams);
    vector<int> indicesBG(1);
    vector<float> distsBG(1);
    flann::SearchParams params(128);

#if 1
    flann::Index kdtreeFG(cv::Mat(_foreground).reshape(1), indexParams);
    vector<int> indicesFG(1);
    vector<float> distsFG(1);
    int counter=0;


    for(int i=0; i<_image.rows; i++)
      {
      for(int j=0; j<_image.cols; j++)
	{

	  vector<float> query(3);
	  query[0]=_imageLab.at<cv::Vec3b>(i,j)[0];
	  query[1]=_imageLab.at<cv::Vec3b>(i,j)[1];
	  query[2]=_imageLab.at<cv::Vec3b>(i,j)[2];
	  kdtreeBG.knnSearch(query, indicesBG, distsBG, 1,params);
	  kdtreeFG.knnSearch(query, indicesFG, distsFG, 1,params);


	  
	  rgbdisBG = std::sqrt(distsBG[0]);
	  rgbdisFG = std::sqrt(distsFG[0]);

	  //	  if(rgbdisBG>0)



	  for (int k=0; k<numLabels; k++)
	    {
	      int pixelindex = i*_image.cols + j;

	      

	      if (k==0)//rgb only
		{
		data[pixelindex*numLabels+k] = rgbdisBG/(rgbdisFG+rgbdisBG);	
		}
	      else
		{
		data[pixelindex*numLabels+k] = rgbdisFG/(rgbdisFG+rgbdisBG);
		}
	      
	    }
	}
      }//end of for(int i=0)

    cout<<"counter: "<<counter<<endl;;
    float *smooth = new float[numLabels*numLabels];
    smooth[0] = 0.;
    smooth[1] = 1;
    smooth[2] = 1;
    smooth[3]=0.;

    try{
      GCoptimizationGeneralGraph *gc = new GCoptimizationGeneralGraph(numPixels,numLabels);

      gc->setDataCost(data);
      gc->setSmoothCost(smooth);
      float Imgvar=calImgVar(_image);

      for(int i=0; i<_image.rows-1; i++)
      {
	for(int j=0; j<_image.cols-1; j++)
	{
	  int p=i*_image.cols+j;
	  int right=i*_image.cols+j+1;
	  int bottom = (i+1)*_image.cols+j;
	  float Disrgb_ij = rgbDis(_imageLab.at<cv::Vec3b>(i,j),_imageLab.at<cv::Vec3b>(i+1,j));
	  double Gij=cv::exp(-(Disrgb_ij* Disrgb_ij)/(2*Imgvar));
          gc->setNeighbors(p,bottom,Gij);
	  Disrgb_ij = rgbDis(_imageLab.at<cv::Vec3b>(i,j),_imageLab.at<cv::Vec3b>(i,j+1));
	  Gij=cv::exp(-(Disrgb_ij* Disrgb_ij)/(2*Imgvar));
          gc->setNeighbors(p,right,Gij);
	}
      }

      
      cout<<"Before optimization energy is"<<gc->compute_energy()<<endl;
      gc->swap(2);
      cout<<"after optimization energy is"<<gc->compute_energy()<<endl;
      
      for(int i=0; i< numPixels; i++)
	{
	  result[i] = gc->whatLabel(i);
	}
      
      delete gc;
    }
    catch(GCException e)
      {
	e.Report();
      }
    
    Mat segmented(_image.rows,_image.cols,CV_8UC3,cv::Scalar(0,0,0));
    for(int i=0; i<_image.rows; i++)
      {
	for(int j=0; j<_image.cols; j++)
	{
	  if(result[i*_image.cols+j]==1)
	    segmented.at<Vec3b>(i,j)=_image.at<Vec3b>(i,j);
	}
      }

#endif
    imshow("mask2", segmented);
    
  }
  
  void mouseClick(int event, int x, int y, int flags, void*)
  {
    switch(event)
      {
      case EVENT_LBUTTONDOWN:  //set foreground;
	{
	  float row = y;
	  float col = x;
	  _foreground.push_back(cv::Vec3f(_imageLab.at<cv::Vec3b>(row,col)[0],
					  _imageLab.at<cv::Vec3b>(row,col)[1],
					  _imageLab.at<cv::Vec3b>(row,col)[2]));
	  cout<<_foreground.size()<<endl;
	}
	break;
      case EVENT_RBUTTONDOWN:  //set background;
	{
	  float row = y;
	  float col = x;
	  _background.push_back(cv::Vec3f(_imageLab.at<cv::Vec3b>(row,col)[0],
					  _imageLab.at<cv::Vec3b>(row,col)[1],
					  _imageLab.at<cv::Vec3b>(row,col)[2]));
	  cout<<_background.size()<<endl;
	}
	break;

      }
  } 
};

rgbdSegment rs;

void on_mouse(int event, int x, int y, int flags, void* param)
{
  rs.mouseClick(event,x,y,flags,param);
}


int main(int argc, char** argv)
{
  if(argc<2)
    {
      cout<<"please input file name"<<endl;
      return 0;
    }
  Mat src = imread(argv[1]);
  rs.init(src);
  const string winName = "image";
  namedWindow( winName, WINDOW_AUTOSIZE );
  setMouseCallback( winName, on_mouse, 0 );
  imshow("image",src);
  while(1)
    {
  char c = waitKey(0);
    if(c=='n')
      {
      rs.process();
      }
    }
  return 0;
}
