/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/


#ifndef ORBMATCHER_H
#define ORBMATCHER_H

#include<vector>
#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>
using namespace std;
using namespace cv;



namespace orb_matcher{
class CV_EXPORTS_W ORBmatcher
{    
#define FRAME_GRID_ROWS 48
#define FRAME_GRID_COLS 64
public:

    CV_WRAP ORBmatcher(int wid, int hei, float nnratio=0.75, bool checkOri=true);

    // Computes the Hamming distance between two ORB descriptors
    static int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);
    static float DescriptorDistanceSIFT(const cv::Mat &a, const cv::Mat &b);

    CV_WRAP int find_matches(cv::InputArray _k1, cv::InputArray _k2,cv::Mat mDescriptors1, cv::Mat mDescriptors2,  int windowSize, int type, OutputArray img2_coordinates);
    bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);
    void AssignFeaturesToGrid(std::vector<cv::KeyPoint> mvKeysUn);
    std::vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel, std::vector<cv::KeyPoint> kps2) const;



public:

    static const int TH_LOW;
    static const int TH_HIGH;
    static const int HISTO_LENGTH;
    int N ;
    float mfGridElementWidthInv;
    float mfGridElementHeightInv;
    std::vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];
    std::vector<cv::KeyPoint> mvKeysUn;
    float mnMinX = 0, mnMinY = 0;
    float mnMaxX , mnMaxY;



    void ComputeThreeMaxima(std::vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3);

    float mfNNratio;
    bool mbCheckOrientation;
};
}

#endif // ORBMATCHER_H
