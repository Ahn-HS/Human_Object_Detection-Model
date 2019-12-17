#include "tracker.h"

Tracker::Tracker(int videoWidth, int videoHeight)
{
    this->videoWidth = videoWidth;
    this->videoHeight = videoHeight;
}



void Tracker::createFeatures(cv::Mat sequence, std::vector<cv::Rect> pedestriansSubDetected, std::vector<BlobPedestrian> &blobPedestrian)
{
    if(pedestriansSubDetected.size() == 0)
        return;

    cv::Mat sequenceGray;
    int maxCorners = 25;
    double qualityLevel = 0.01;
    double minDistance = 1.;
    int blockSize = 3;
    bool useHarrisDetector = false;
    double kdef = 0.04;

    cv::cvtColor(sequence, sequenceGray, CV_BGR2GRAY);

    blobPedestrian.clear();
    blobPedestrian.resize(pedestriansSubDetected.size());

    for(size_t j=0;j<blobPedestrian.size();j++)
    {
        cv::goodFeaturesToTrack(sequenceGray(pedestriansSubDetected[j]), blobPedestrian[j].features, maxCorners, qualityLevel, minDistance, cv::noArray(), blockSize, useHarrisDetector, kdef);

        for(size_t k=0;k<blobPedestrian[j].features.size();k++)
        {
            blobPedestrian[j].features[k].x += pedestriansSubDetected[j].x;
            blobPedestrian[j].features[k].y += pedestriansSubDetected[j].y;
        }

        blobPedestrian[j].window = cv::boundingRect(blobPedestrian[j].features);
    }
}



void Tracker::trackFeatures(cv::Mat previousSequence, cv::Mat sequence, std::vector<BlobPedestrian> &blobPedestrian)
{
    if(blobPedestrian.size() == 0)
        return;

    cv::Mat previousSequenceGray;
    cv::Mat sequenceGray;
    std::vector<cv::Point2f> newFeatures;
    std::vector<uchar> status;
    std::vector<float> err;

    cv::cvtColor(previousSequence, previousSequenceGray, CV_BGR2GRAY);
    cv::cvtColor(sequence, sequenceGray, CV_BGR2GRAY);

    for(size_t j=0;j<blobPedestrian.size();j++)
    {
        cv::calcOpticalFlowPyrLK(previousSequenceGray, sequenceGray, blobPedestrian[j].features, newFeatures, status, err);

        blobPedestrian[j].features.clear();
        blobPedestrian[j].features = newFeatures;
        blobPedestrian[j].window = cv::boundingRect(blobPedestrian[j].features);

        newFeatures.clear();
    }
}



void Tracker::camshift(std::vector<Pedestrian> &pedestrian)
{
    for(size_t i=0;i<pedestrian.size();i++)
    {
        if(pedestrian[i].window.x == 0)
            pedestrian.erase(pedestrian.begin()+i);

        else if(pedestrian[i].window.y == 0)
            pedestrian.erase(pedestrian.begin()+i);

        else if((pedestrian[i].window.x+pedestrian[i].window.width) == this->videoWidth)
            pedestrian.erase(pedestrian.begin()+i);

        else if((pedestrian[i].window.x+pedestrian[i].window.width) == this->videoWidth)
            pedestrian.erase(pedestrian.begin()+i);

        else
            cv::CamShift(pedestrian[i].backProj, pedestrian[i].window, cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 10, 1));
    }
}
