#include "orb_extractor/src/ORB.hpp"
#include "orb_matcher/src/ORB_matcher.hpp"
#include "opencv2/opencv.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include "iostream"
#define FRAME_GRID_COLS 64
#define FRAME_GRID_ROWS 48


using namespace cv;
using namespace std;
using namespace orb;
using namespace orb_matcher;

int main()
{
    cv::Mat mDescriptors1, mDescriptors2, imout1,imout2, out3 ;
    cv::Mat mvKeys1, mvKeys2;
    ORBextractor mpIniORBextractor = ORBextractor(2048,1.2,8,20,5);

    for(int imgiterator = 124 ;imgiterator<826;imgiterator+=1)
    {
        
        std::string img1str = "/home/olorin/Downloads/IISC_Road_data/cds_dept/1.bag/left_image/frame0";
        img1str.append(std::to_string(imgiterator));
        img1str.append(".jpg");
        cv::Mat im = cv::imread(img1str);
        cv::cvtColor(im, im,  cv::COLOR_BGR2GRAY);
        // cv::cvtColor(im, im, CV_BGR2GRAY);
        std::string img2str = "/home/olorin/Downloads/IISC_Road_data/cds_dept/1.bag/left_image/frame0";
        img2str.append(std::to_string(imgiterator+1));
        img2str.append(".jpg");
        cv::Mat im2 = cv::imread(img2str);
        cv::cvtColor(im2, im2,  cv::COLOR_BGR2GRAY);
        int64 t0 = cv::getTickCount();
        mpIniORBextractor.extract_orb_fts(im,cv::Mat(),mvKeys1,mDescriptors1);
        int64 t1 = cv::getTickCount();
        mpIniORBextractor.extract_orb_fts(im2,cv::Mat(),mvKeys2,mDescriptors2);
        // cv::Mat kpx1 = mvKeys1.getMat();
        // cv::Mat kpx2 = mvKeys2.getMat();



        // (*mpIniORBextractor)(im,cv::Mat(),mvKeys1,mDescriptors1);
        // (*mpIniORBextractor)(im2,cv::Mat(),mvKeys2,mDescriptors2);

        vector<int> matches12;

        // std::cout<<mvKeys1.size()<<"-->"<<mvKeys2.size();

        ORBmatcher ob = ORBmatcher();
        int64 t2 = cv::getTickCount();
        int mtchs = ob.find_matches(mvKeys1, mvKeys2,mDescriptors1, mDescriptors2, 100, matches12);
        int64 t3 = cv::getTickCount();
        std::cout<<"FILE"<<" "<<imgiterator<<" "<<imgiterator+1<<" "<<mtchs<<"\n";
        vector<cv::DMatch> dmatches;
        for(int i = 0; i<matches12.size();i++)
        {
            // std::cout<<matches12[i]<<"_";
            if(matches12[i]!=-1)
            {
                cv::DMatch dm(i, matches12[i], 1);
                dmatches.push_back(dm);
            }
        }
        // cv::Mat myks;
        // mvKeys1.copyTo(myks);
        // cv::sort(mvKeys1, myks, CV_SORT_EVERY_ROW + CV_SORT_ASCENDING);

        // std::cout<<myks;
        // std::cout<<
        // std::cout<<imgiterator<<" "<<mtchs;
        std::vector<cv::Point2f> input1;
        for (int x = 0; x < mvKeys1.rows; x++){
                // std::cout<<mvKeys1.at<double>(x,0);
                input1.push_back(cv::Point2f(mvKeys1.at<cv::Point2f>(x)));
            }

        std::vector<cv::Point2f> input2;
        for (int x = 0; x < mvKeys2.rows; x++)
                input2.push_back(cv::Point2f(mvKeys2.at<cv::Point2f>(x)));

        vector<KeyPoint> kps1, kps2;
        cv::KeyPoint::convert(input1, kps1);
        cv::KeyPoint::convert(input2, kps2);
        std::vector<char> mask(mtchs, 1);

        // // std::vector<char> mask(mtchs, 1);
        
        cv::drawMatches(im,kps1, im2, kps2, dmatches, out3, cv::Scalar::all(-1), cv::Scalar::all(-1), mask, DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        
        // cv::drawKeypoints(im, kps1, imout1);
        // cv::drawKeypoints(im2, kps2, imout2);
        // cv::imshow("Image1", imout1);
        // cv::imshow("Image2", imout2);
        cv::imshow("Out", out3);
        cv::waitKey(1);
    }
    cv::destroyAllWindows();

    // std::cout<<mDescriptors;
    return -1;
}