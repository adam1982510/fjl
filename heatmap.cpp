#include <iostream>  
#include <opencv2/opencv.hpp>  
#include <opencv2/video/background_segm.hpp>


using namespace std;
using namespace cv;

typedef struct{
		Point pt;
		int  flowcnt;
}sPoint;
//对轮廓按面积降序排列  
bool biggerSort(vector<Point> v1, vector<Point> v2)
{
	return contourArea(v1) > contourArea(v2);
}

bool biggersPointSort(sPoint pt1, sPoint pt2)
{
	return pt1.flowcnt > pt2.flowcnt;
}

Point getRectCenter(Rect rect)
{
	Point pt;
	pt.x = rect.x + cvRound(rect.width / 2.0);
	pt.y = rect.y + (rect.height / 2.0);
	return pt;
}

int getDistancePP(Point p1, Point p2)
{
	int distance = 0;
	distance = powf((p1.x - p2.x), 2) + powf((p1.y - p2.y), 2);
	distance = sqrtf(distance);
	return distance;
}

Mat frame, frame1, frame_black, im_color, heatmap;          //当前帧  

int g_flowDistanceThd = 50;  //距离阈值
int g_flowCountThd = 60;	 //密度阈值
int g_frameFps = 25;		 //每秒帧率
int g_frameMaxCount = 120; //120/25 = 5s
int g_frameCountThd = 100; //100/25 = 4s
int g_resolution_w = 960;
int g_resolution_h = 512;

double g_alpha = 0.3;
double g_beta = 1 - g_alpha; 
int g_threshval = g_alpha * g_resolution_w;
string g_orginfilename;
string g_heatMapfilename;
string g_heatMapBackfilename;
string g_heatMapDatafilename;

static void on_AlphaTrackBar(int thd, void*)
{
	g_alpha = thd / 255.0;
	g_beta = 1 - g_alpha;
	addWeighted(im_color, g_alpha, frame, g_beta, 0, heatmap);
}

int main()
{
	//视频不存在，就返回  
	//VideoCapture cap("rtsp://admin:heyubin1215@172.21.77.98//Streaming/Channels/102");
	//VideoCapture cap("d:/vtest.avi");
	//VideoCapture cap("rtsp://admin:heyubin1215@172.21.77.98//Streaming/Channels/102");
	//VideoCapture cap("d:/Camera2_20170613_142553");
	//VideoCapture cap("d:/Camera2_20170614_161550");
	//VideoCapture cap("d:/Camera2_20170621_144101");
	string filename = "Camera2_20170621_144101";
	string path = "d:/" + filename;
	VideoCapture cap(path);
	if (cap.isOpened() == false)
		return 0;

	g_orginfilename = path + ".jpg";
	g_heatMapfilename = path + "-mg.jpg";
	g_heatMapBackfilename = path + "-bg.jpg";
	g_heatMapDatafilename = path + "-data.txt";
	

	//定义变量  
	int i;

	//Mat frame, frame1,  frame_black, im_color, heatmap;          //当前帧  
	Mat frame_br(g_resolution_h, g_resolution_w, CV_8U, Scalar(0));
	Mat foreground;     //前景  
	Mat bw;             //中间二值变量  
	Mat se;             //形态学结构元素  
	vector<Point> hisFlowPoints;
	vector<vector<Point>> hisFlowPointsVector;
	vector<sPoint> tempFlowPoints;
	vector<sPoint> hisValidFlowPoints;   //有效人流记录


	//bool reverseFlag = false;
	//用混合高斯模型训练背景图像  
	//BackgroundSubtractorMOG2 mog ;
	Ptr<BackgroundSubtractorMOG2> mog = createBackgroundSubtractorMOG2();
	for (i = 0; i < 10; ++i)
	{
		cout << "正在训练背景:" << i << endl;
		cap >> frame1;
		if (frame1.empty() == true)
		{
			cout << "视频帧太少，无法训练背景" << endl;
			getchar();
			return 0;
		}
		resize(frame1, frame, Size(g_resolution_w, g_resolution_h), 0, 0, CV_INTER_LINEAR);
		mog->apply(frame, foreground, 0.01);
	}
	imwrite(g_orginfilename, frame);
	//frame.copyTo(frame_br);
	//frame_br = Scalar::all(255);
	frame.copyTo(heatmap);
	frame.copyTo(frame_black);
	frame_br.copyTo(im_color);
	//im_color = Scalar::all(255);

	//trackerbar define
	//热度图透明度
	namedWindow("heatmap", 1);
	createTrackbar("alphaTb", "heatmap", &g_threshval, g_resolution_w, on_AlphaTrackBar);

	namedWindow("跟踪效果", 1);
	createTrackbar("flowDistanceThd", "跟踪效果", &g_flowDistanceThd, 255, NULL);
	createTrackbar("flowCountThd", "跟踪效果", &g_flowCountThd, 255, NULL);


	//rectangle(frame_black,  );
	//目标外接框、生成结构元素（用于连接断开的小目标）  
	Rect rt, rt_temp;
	se = getStructuringElement(MORPH_RECT, Size(10, 10));

	//统计目标直方图时使用到的变量  
	vector<Mat> vecImg;
	vector<int> vecChannel;
	vector<int> vecHistSize;
	vector<float> vecRange;
	Mat mask(frame.rows, frame.cols, DataType<uchar>::type);
	//变量初始化  
	vecChannel.push_back(0);
	vecHistSize.push_back(32);
	vecRange.push_back(0);
	vecRange.push_back(180);
	frame_br = Scalar::all(255);

	Mat hsv;        //HSV颜色空间，在色调H上跟踪目标（camshift是基于颜色直方图的算法）  
	MatND hist;     //直方图数组  
	double maxVal;      //直方图最大值，为了便于投影图显示，需要将直方图规一化到[0 255]区间上  
	Mat backP;      //反射投影图  
	Mat result;     //跟踪结果  

	//视频处理流程 
	int frameCnt = 0;
	int emptyCnt = 0;
	while (1)
	{
		double t = (double)getTickCount();
		//读视频  
		cap >> frame1;
		if (frame1.empty() == true)
			break;
		frameCnt++;
		if (frameCnt % 2 == 0)
		{
		//	continue;
		}
		
		resize(frame1, frame, Size(g_resolution_w, g_resolution_h), 0, 0, CV_INTER_LINEAR);
		//生成结果图  
		frame.copyTo(result);
		imshow("原图", frame);
		moveWindow("原图", 0, 0);

		//检测目标(其实是边训练边检测)  
		mog->apply(frame, foreground, 0.01);
		imshow("混合高斯检测前景", foreground);
		moveWindow("混合高斯检测前景", 600, 0);
		//对前景进行中值滤波、形态学膨胀操作，以去除伪目标和接连断开的小目标       
		medianBlur(foreground, foreground, 3);
		imshow("中值滤波", foreground);
		moveWindow("中值滤波", 1100, 0);
		morphologyEx(foreground, foreground, MORPH_DILATE, se);
		imshow("中值滤波2", foreground);
		moveWindow("中值滤波2", 1100, 500);
		//检索前景中各个连通分量的轮廓  
		foreground.copyTo(bw);
		vector<vector<Point>> contours;
		findContours(bw, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
		if (contours.size() > 0)
		{
			/*	int maxAreaIdx = -1, iteratorIdx = 0;//n为面积最大轮廓索引，m为迭代索引
			for (int k = 0; k < contours.size(); ++k)
			{

				double tmparea = fabs(contourArea(contours[k]));
				if (tmparea > maxarea)
				{
					maxarea = tmparea;
					maxAreaIdx = iteratorIdx;
					continue;
				}
				if (tmparea < minarea)
				{
					//删除面积小于设定值的轮廓  
					cvSeqRemove(contours, 0);
					continue;
				}
				CvRect aRect = boundingRect(contours[k]);
				if (aRect.width > 60 && aRect.height < 260)
				{
					continue;
				}
				else
				{
					//删除宽高比例小于设定值的轮廓  
					cvSeqRemove(&contours, k);
					continue;
				}

			}*/

			//continue;
		//对连通分量进行排序  
			std::sort(contours.begin(), contours.end(), biggerSort);

			//结合camshift更新跟踪位置（由于camshift算法在单一背景下，跟踪效果非常好；  
			//但是在监控视频中，由于分辨率太低、视频质量太差、目标太大、目标颜色不够显著  
			//等各种因素，导致跟踪效果非常差。  因此，需要边跟踪、边检测，如果跟踪不够好，  
			//就用检测位置修改  
			cvtColor(frame, hsv, COLOR_BGR2HSV);
			//imshow("hsv", hsv);
			//moveWindow("hsv", 0, 500);
			vecImg.clear();
			vecImg.push_back(hsv);
			for (int k = 0; k < contours.size(); ++k)
			{
				//第k个连通分量的外接矩形框  
				//if (contourArea(contours[k]) < contourArea(contours[0]) / 5)
				//	break;
				rt = boundingRect(contours[k]);
				if (rt.width < 50)
					continue;
				
				mask = 0;
				mask(rt) = 255;

				//统计直方图  
				calcHist(vecImg, vecChannel, mask, hist, vecHistSize, vecRange);
				minMaxLoc(hist, 0, &maxVal);
				hist = hist * 255 / maxVal;
				//计算反向投影图  
				calcBackProject(vecImg, vecChannel, hist, backP, vecRange, 1);
				imshow("反向投影", backP);
				moveWindow("反向投影", 600, 500);
				//camshift跟踪位置  
				Rect search = rt;
				rt_temp = rt;
				RotatedRect rrt = CamShift(backP, search, TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 10, 1));
				Rect rt2 = rrt.boundingRect();
				//printf("contours[%d]: w1=%d, h1=%d, x=%d, y=%d, w2=%d, h2=%d, x2=%d, y2=%d, framenum=%d\n", k, rt.width, rt.height, rt.x, rt.y, rt2.width, rt2.height, rt2.x, rt2.y, frameCnt);
				rt &= rt2;

				//跟踪框画到视频上  
				
				//if ((rt.width > 60 && rt.width < 120)
				//	&& rt.height > 60 && rt.height < 300)
				{	
					Scalar color = Scalar(0, 50*k, k * 50);
					rectangle(result, rt, color, 2);
					color = Scalar(0, 255, 255);
					rectangle(result, rt_temp, color, 1);
					Point center = getRectCenter(rt);
					center.y = rt.y + g_flowDistanceThd/2;
					circle(result, center, g_flowDistanceThd/2, color);
					char tmpstr[60];
					sprintf(tmpstr, "%d", rt.width);
					putText(result, tmpstr, Point(rt.x, rt.y), CV_FONT_HERSHEY_DUPLEX, 1.0f, CV_RGB(0, 255, 255));

					//record point
				/*	if (reverseFlag == false)
					{
						hisFlowPoints.push_back(center);
					}
					else
					{
						if (hisFlowPoints.size() > 0)
						{
							hisFlowPoints.erase(hisFlowPoints.begin());
							hisFlowPoints.push_back(center);
						}
						else
						{
							reverseFlag = false;
						}
						
					}*/
					hisFlowPoints.push_back(center);
				}
			}

			

			//结果显示  
			//imshow("原图", frame);
			//moveWindow("原图", 0, 0);

		//	imshow("膨胀运算", foreground);
		//	moveWindow("膨胀运算", 0, 350);

		//	imshow("反向投影", backP);
		//	moveWindow("反向投影", 400, 350);

		//	imshow("跟踪效果", result);
		//	moveWindow("跟踪效果", 0, 0);
		//	imshow("跟踪效果bk", frame_br);
		//	moveWindow("跟踪效果bk", 600, 500);
		}
		else
		{
			emptyCnt++;
			if (emptyCnt > g_frameCountThd)   //20/25帧 * 5s 
			{
				for (i = 0; i < 10; ++i)
				{
					cout << "空闲训练背景:" << i << endl;
					cap >> frame1;
					if (frame1.empty() == true)
					{
						cout << "视频帧太少，无法训练背景" << endl;
						getchar();
						return 0;
					}
					resize(frame1, frame, Size(g_resolution_w, g_resolution_h), 0, 0, CV_INTER_LINEAR);
					mog->apply(frame, foreground, 0.01);
				}
				emptyCnt = 0;
				//hisFlowPoints.clear();
			}
			
		}
		//储存连续g_frameMaxCount/25 时间数据
		if (hisFlowPointsVector.size() < g_frameMaxCount)
		{
			hisFlowPointsVector.push_back(hisFlowPoints);
		}
		else
		{
			hisFlowPointsVector.push_back(hisFlowPoints);
			hisFlowPointsVector.erase(hisFlowPointsVector.begin());
		}
		hisFlowPoints.clear();

		//滞留时间间隔 默认4s处理一次 g_frameCountThd/25=4s 
		
		//if ((frameCnt % 100) == 0 && (hisFlowPoints.size()) > 0)
		if ((frameCnt % g_frameCountThd) == 0 && (hisFlowPointsVector.size() >= g_frameMaxCount))
		{
			frame_black = Scalar::all(0);
			int i = 0, j = 0, i1 = 0, j1 = 0, distancePP = 0, flowCnt = 0, psize = 0;
			sPoint pt1, pt2;

			for (i = 0; i < hisFlowPointsVector.size(); i++)
			{
				psize = hisFlowPointsVector.at(i).size();
				for (i1 = 0; i1 < hisFlowPointsVector.at(i).size(); i1++)
				{
					pt1.pt = hisFlowPointsVector.at(i).at(i1);
					flowCnt = 0;
					for (j = 0; j < hisFlowPointsVector.size(); j++)
					{
						for (j1 = 0; j1 < hisFlowPointsVector.at(j).size(); j1++)
						{
							pt2.pt = hisFlowPointsVector.at(j).at(j1);
							distancePP = getDistancePP(pt1.pt, pt2.pt);
							if (distancePP < g_flowDistanceThd)
							{
								flowCnt++;
							}
						}

					}
					if (flowCnt > g_flowCountThd) //大于人流热点阈值  50 / 25帧/s = 2.5s
					{
						pt1.flowcnt = flowCnt;
						tempFlowPoints.push_back(pt1);
						//	circle(frame_br, pt1.pt, 2, Scalar(0, 255, 255));
						circle(result, pt1.pt, g_flowDistanceThd, Scalar(0, 255, 0));
						circle(frame_black, pt1.pt, 1, Scalar(0, 255, 0));
						/*	char snow[50] = {0};
						time_t now = time(0);

						strftime(snow, 50,"%H:%M:%S",localtime(&now));
						printf("hisValidFlowPoints vsize=%d hsize=%d flowCnt=%d time=%s\n", hisValidFlowPoints.size(), hisFlowPoints.size(), flowCnt, snow);
						*/
					}
				}
			}


			//有效points再聚类及过滤过程
			psize = tempFlowPoints.size();
			if (psize > 0)
			{
				std::sort(tempFlowPoints.begin(), tempFlowPoints.end(), biggersPointSort);
				while (tempFlowPoints.size() > 0)
				{
					//std:copy(tempFlowPoints.begin(), tempFlowPoints.end(), std::back_inserter(temp1FlowPoints));
					psize = tempFlowPoints.size();
					pt1 = tempFlowPoints.at(0);
					if (psize > 1)
					{
						vector <sPoint>::iterator iter = tempFlowPoints.begin() + 1;
						for (; iter != tempFlowPoints.end();)
						{
							pt2 = *iter;
							distancePP = getDistancePP(pt1.pt, pt2.pt);
							if (distancePP < g_flowDistanceThd)
							{
								iter = tempFlowPoints.erase(iter);
							}
							else
							{
								iter++;
							}
						}
					}

					hisValidFlowPoints.push_back(pt1);
					tempFlowPoints.erase(tempFlowPoints.begin());
					circle(frame_br, pt1.pt, 1, Scalar(0, 255, 255), -1);
					char tmpstr[60];
					sprintf(tmpstr, "%d", hisValidFlowPoints.size());
					putText(frame_br, tmpstr, Point(pt1.pt.x, pt1.pt.y), CV_FONT_HERSHEY_DUPLEX, 0.5f, CV_RGB(0, 255, 255));

					char snow[50] = { 0 };
					time_t now = time(0);
					strftime(snow, 50, "%H:%M:%S", localtime(&now));
					printf("hisValidFlowPoints vsize=%d hsize=%d flowCnt=%d time=%s\n", hisValidFlowPoints.size(), tempFlowPoints.size(), flowCnt, snow);

					ofstream  fileheatmap(g_heatMapDatafilename, ofstream::app);
					if (fileheatmap.is_open())
					{
						char tdata[50] = {0};
						tdata[49] = '\n';
						sprintf(tdata, "%d %d %s", pt1.pt.x, pt1.pt.y, snow);
						fileheatmap.write(tdata, sizeof(tdata));
						fileheatmap.close();
					}
				}
				//frame_br.convertTo(frame_br, CV_8U, 10);
				distanceTransform(frame_br, im_color, CV_DIST_L2, 5);
				im_color.convertTo(im_color, CV_8U, 30);
				applyColorMap(im_color, im_color, COLORMAP_RAINBOW);
				addWeighted(im_color, g_alpha, frame, g_beta, 0, heatmap);
				imwrite(g_heatMapfilename, heatmap);
				imwrite(g_heatMapBackfilename, im_color);
				
			}

		}
		imshow("跟踪效果", result);
		moveWindow("跟踪效果", 0, 0);
		imshow("heatmap_back", im_color);
		moveWindow("heatmap_back", g_resolution_w, 500);
		imshow("density_live", frame_black);
		moveWindow("density_live", 0, 500);
		imshow("heatmap", heatmap);
		moveWindow("heatmap", g_resolution_w, 0);
		t = (double)getTickCount() - t;
		//cout << "load_images pos images end!." << "size:" << frameCnt << " cost time = " << (t*1000. / cv::getTickFrequency()) << " ms" << endl;
		waitKey(10);
	}

	getchar();
	return 0;
}