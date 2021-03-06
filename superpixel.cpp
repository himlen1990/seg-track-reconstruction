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
  std::vector<cv::Vec2f> _superPixelXY, _foreground;
  std::vector<cv::Vec3f> _superPixelRGB;


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
    getSuperPixels(_imageLab,_labels, _superPixelXY, _superPixelRGB);//, meansdepth);

  }


  float rgbDis(cv::Vec3f &v1,cv::Vec3f &v2)
  {
    return  cv::norm(v1,v2,cv::NORM_L2);
  }


  float depthDis(float d1, float d2)
  {
    return  cv::abs(d1-d2);
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

  void getSuperPixels(const cv::Mat &Img, const cv::Mat &labels, std::vector<cv::Vec2f> &superPixelXY, std::vector<cv::Vec3f> &superPixelRGB){// std::vector<float> &meansdepth) {
    //rename to superpixel processing
    // Count superpixels or get highest superpixel index:
    int superpixels = 0;

    Ptr<cv::ximgproc::SuperpixelSLIC> slic = cv::ximgproc::createSuperpixelSLIC(Img,cv::ximgproc::SLICO, _region_size);
    slic->iterate();
    slic->enforceLabelConnectivity();
    slic->getLabels(labels);
    slic->getLabelContourMask(_maskfordebug);
    int number = slic->getNumberOfSuperpixels();

    //imshow("mask",_maskfordebug);
    cout<<"number of superpixels: "<<number<<endl;
    for (int i = 0; i < labels.rows; ++i) {
      for (int j = 0; j < labels.cols; ++j) {
	if (labels.at<int>(i, j) > superpixels) {
	  superpixels = labels.at<int>(i, j);
	}
      }
    }

    superpixels++;

    // Setup means as zero vectors.
    superPixelXY.clear();

    superPixelXY.resize(superpixels);
    superPixelRGB.clear();
    superPixelRGB.resize(superpixels);
    //meansdepth.clear();
    //meansdepth.resize(superpixels);


    for (int k = 0; k < superpixels; k++)
      {
	superPixelXY[k] = cv::Vec2f(0, 0);
	superPixelRGB[k] = cv::Vec3f(0, 0,0);
	//meansdepth[k] = 0;
      }

    std::vector<int> counts(superpixels, 0);

    // Sum y and x coordinates for each superpixel:
    for (int i = 0; i < labels.rows; ++i) {
      for (int j = 0; j < labels.cols; ++j) {
	superPixelXY[labels.at<int>(i, j)][0] += i; // for computing mean i (i.e. row or y axis)
	superPixelXY[labels.at<int>(i, j)][1] += j; // for computing the mean j (i.e. column or x axis)

	superPixelRGB[labels.at<int>(i, j)][0] += Img.at<cv::Vec3b>(i, j)[0];
	superPixelRGB[labels.at<int>(i, j)][1] += Img.at<cv::Vec3b>(i, j)[1];
	superPixelRGB[labels.at<int>(i, j)][2] += Img.at<cv::Vec3b>(i, j)[2];
	//meansdepth[labels.at<int>(i, j)][5] += depthImg.at<cv::32FC1>(i, j)[0];

	counts[labels.at<int>(i, j)]++;
      }
    }

    // Obtain averages by dividing by the size (=number of pixels) of the superpixels.
    for (int k = 0; k < superPixelXY.size(); ++k) {
      superPixelXY[k] /= counts[k];
      superPixelRGB[k] /= counts[k];    
      //meansdepth[k] /= counts[k];
    }
  }

  void getPotentialBG(std::vector<cv::Vec2f> &superPixelXY, std::vector<cv::Vec2f> &bgCandidates, std::vector<int> &candidateIndex )
  {
    bgCandidates.clear();
    for (int k = 0; k < superPixelXY.size(); ++k) {
      if(//(superPixelXY[k][0]<_MaxboundaryMargin)||
	 (superPixelXY[k][1]<_MaxboundaryMargin)||
	 // ((_imgHeight-superPixelXY[k][0])<_MaxboundaryMargin)||
	 ((_imgWidth-superPixelXY[k][1])<_MaxboundaryMargin))
	{
	  candidateIndex.push_back(k);//we need a index to access the rgb and depth value
	  bgCandidates.push_back(superPixelXY[k]);      
	}
    }
  }
  


  void process()
  {
    std::vector<cv::Vec2f> _bgCandidates; 
    std::vector<float> _superPixelDepth;
    std::vector<int> _candidateIndex;

    getPotentialBG(_superPixelXY,_bgCandidates, _candidateIndex);

    cout<<_bgCandidates.size()<<endl;
    flann::KDTreeIndexParams indexParams(2);
    flann::Index kdtreeBG(cv::Mat(_bgCandidates).reshape(1), indexParams);
    vector<int> indicesBG(1);
    vector<float> distsBG(1);
    flann::SearchParams params(128);


    //build gco graph
    int cues = 1; //RGB and depth
    int numLabels = 2*cues;
    int numPixels = _superPixelXY.size();
    int *result = new int[numPixels];
    float *data = new float[numPixels*numLabels];
    float rgbdisBG;
    float rgbdisFG;
    if(_foreground.size()==0)
      {
	cout<<"we need a foreground"<<endl;
	return;
      }

    flann::Index kdtreeFG(cv::Mat(_foreground).reshape(1), indexParams);
    vector<int> indicesFG(1);
    vector<float> distsFG(1);

   
    //unary term
    for(int i=0; i<_superPixelXY.size(); i++)
      {

	vector<float> query(2);
	query[0]=_superPixelXY[i][0];
	query[1]=_superPixelXY[i][1];
	kdtreeBG.knnSearch(query, indicesBG, distsBG, 1,params);
	kdtreeFG.knnSearch(query, indicesFG, distsFG, 1,params);
	cv::Vec3f foregroundRGB; //we don't use superpixel's RGB value for foreground, use raw pixel directly 

	foregroundRGB[0] = _imageLab.at<cv::Vec3b>(_foreground[indicesFG[0]][0],_foreground[indicesFG[0]][1])[0] ;
	foregroundRGB[1] = _imageLab.at<cv::Vec3b>(_foreground[indicesFG[0]][0],_foreground[indicesFG[0]][1])[1];
	foregroundRGB[2] = _imageLab.at<cv::Vec3b>(_foreground[indicesFG[0]][0],_foreground[indicesFG[0]][1])[2];


	rgbdisBG = rgbDis(_superPixelRGB[i],_superPixelRGB[_candidateIndex[indicesBG[0]]]);
	rgbdisFG = rgbDis(_superPixelRGB[i], foregroundRGB);
	//cout<<rgbdisFG<<endl;
	for (int j=0; j<numLabels; j++)
	  {
	    if (j==0)//rgb only
	      data[i*numLabels+j] = rgbdisBG/(rgbdisFG+rgbdisBG);
	    else
	      data[i*numLabels+j] = rgbdisFG/(rgbdisFG+rgbdisBG);
	  }
	/*
	  cout<<"indices "<<indices[0]<<endl;
	  cout<<"dists "<<dists[0]<<endl;
	  cout<<"meansxy: "<<meansxy[candidateIndex[indices[0]]]<<endl;
	  cout<<"meansxy[i]: "<<meansxy[i]<<endl;
	  cout<<"rgbdis "<<rgbdis<<endl;*/
      }
    

    
    float *smooth = new float[numLabels*numLabels];
    smooth[0] = 0.;
    smooth[1] = 1;
    smooth[2] = 1;
    smooth[3]=0.;
    
    try{
      GCoptimizationGeneralGraph *gc = new GCoptimizationGeneralGraph(numPixels,numLabels);

      gc->setDataCost(data);
      gc->setSmoothCost(smooth);
      
      //find neighbours
      
      flann::Index kdtreePair(cv::Mat(_superPixelXY).reshape(1), indexParams);
      vector<int> indicesPair(5);
      vector<float> distsPair(5);
      
#if 1
      for(int i=0; i<_superPixelXY.size(); i++)
	{
	  if (_superPixelXY[i][0]> (_image.rows-_MaxboundaryMargin) || _superPixelXY[i][1]> (_image.cols-_MaxboundaryMargin)) //do not find neighbour for left and bottom boundary
	    continue;
	  
	  vector<float> query(2);
	  query[0]=_superPixelXY[i][0];
	  query[1]=_superPixelXY[i][1];
	  kdtreePair.knnSearch(query, indicesPair, distsPair, 5 ,params);

	  int maxIndex=0;
	  for(int j=0; j<4; j++)//find bottom neighbour
	    {
	      maxIndex = _superPixelXY[indicesPair[maxIndex]][0]>_superPixelXY[indicesPair[j+1]][0]? maxIndex:j+1;
	    }
	  float Imgvar=calImgVar(_image);
	  float Disrgb_ij = rgbDis(_superPixelRGB[i],_superPixelRGB[indicesPair[maxIndex]]);
	  double Gij=cv::exp(-(Disrgb_ij* Disrgb_ij)/(2*Imgvar));
	  gc->setNeighbors(i,indicesPair[maxIndex],Gij);
	  //cout<<" "<<Gij <<endl;
	  //if have depth
	  //minGij = (-Drgb_ij* Drgb_ij/2*Imgvar)<(-Dd_ij* Dd_ij/2*depImgvar)? a:b
	  //minGij = cv::exp(minGij)
	  maxIndex=0;
	  for(int j=0; j<4; j++)//find right neighbour
	    {
	      maxIndex = _superPixelXY[indicesPair[maxIndex]][1]>_superPixelXY[indicesPair[j+1]][1]? maxIndex:j+1;
	    }
	  Disrgb_ij = rgbDis(_superPixelRGB[i],_superPixelRGB[indicesPair[maxIndex]]);
	  Gij=cv::exp(-(Disrgb_ij* Disrgb_ij)/(2*Imgvar));
	  gc->setNeighbors(i,indicesPair[maxIndex],Gij);	
	  //cout<<"meansxy: "<<meansxy[i]<<endl;
	  //cout<<"indices "<<meansxy[indicesPair[0]]<<meansxy[indicesPair[1]]<<meansxy[indicesPair[2]]<<meansxy[indicesPair[3]]<<meansxy[indicesPair[4]]<<endl;
	  //cout<<"botneighbour "<<meansxy[indicesPair[botmaxIndex]]<<" leftneighbour "<<meansxy[indicesPair[maxIndex]]<<endl;;
	}
#endif
      printf("\nBefore optimization energy is %f",gc->compute_energy());
      gc->swap(2);// run expansion for 2 iterations. For swap use gc->swap(num_iterati \	ons);                                                                                                 
      printf("\nAfter optimization energy is %f",gc->compute_energy());
	
      for(int i=0; i< numPixels; i++)
	{  result[i] = gc->whatLabel(i);
	  //cout<<result[i]<<endl;
	}
      delete gc;
    }
    catch(GCException e)
      {
	e.Report();
      }
  

    //labels.at<int>(i, j)  == index in array meansxy and result//
    Mat segmented(_labels.rows,_labels.cols,CV_8UC3,cv::Scalar(0,0,0));
    for (int i = 0; i < _labels.rows; ++i) {
      for (int j = 0; j < _labels.cols; ++j) {
	if (result[_labels.at<int>(i, j)]==1) //foreground
	  segmented.at<Vec3b>(i,j)=_image.at<Vec3b>(i,j);
      }
    }

    //  cout<<segmented<<endl;
    imshow("mask2", segmented);
    //imshow("image", _image);

  }

  void mouseClick(int event, int x, int y, int flags, void*)
  {
    switch(event)
      {
      case EVENT_LBUTTONDOWN:  //set foreground;
	float row = y;
	float col = x;
	_foreground.push_back(cv::Vec2f(row,col));
	cout<<_foreground.size()<<endl;
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
