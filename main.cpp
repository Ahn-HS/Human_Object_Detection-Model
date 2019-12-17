/**
  * HOG Detection
  *
  **/

#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/features2d/features2d.hpp"

#include <iostream>
#include <pedestrian.h>

enum choiceAlgo{HOG_TEMPLATE_TRACKING,
                HOG_GOODFEATURESTOTRACK_LK,
                OPT_FLOW_FARNEBACK,
                CAMSHIFT_KALMAN_FILTER,
                BACKGROUND_LK,
                BACKGROUND_KALMAN};

enum formatVideo{SEQUENCE_IMAGE,
                 VIDEO,
                 NOT_DEFINED};

enum trackingOption{GOOD_FEATURES_TO_TRACK,
                    LUCAS_KANADE,
                    NOTHING_TO_TRACK};



//------------------METHODES-PARSERS---------------------------------------//

choiceAlgo detectAlgo(std::string argument)
{
    if(argument.compare("template_tracking") == 0)
        return HOG_TEMPLATE_TRACKING;

    else if(argument.compare("LK_tracking") == 0)
        return HOG_GOODFEATURESTOTRACK_LK;

    else if(argument.compare("farneback") == 0)
        return OPT_FLOW_FARNEBACK;

    else if(argument.compare("camshift_kalman") == 0)
        return CAMSHIFT_KALMAN_FILTER;

    else if(argument.compare("background_lk") == 0)
        return BACKGROUND_LK;

    else if(argument.compare("background_kalman") == 0)
        return BACKGROUND_KALMAN;

    else
        return HOG_GOODFEATURESTOTRACK_LK;

}

formatVideo detectFormat(std::string argument)
{
    if(argument.compare("image") == 0)
        return SEQUENCE_IMAGE;

    else if(argument.compare("video") == 0)
        return VIDEO;

    else
        return NOT_DEFINED;
}



void extractVideoData(std::vector<cv::Mat> &sequence, formatVideo format, std::string filePathName, int &nbTrames, double &fps)
{

    if(format == SEQUENCE_IMAGE)
    {
        sequence.resize(nbTrames);
        fps = 15;

        for(int i=0;i<nbTrames;i++)
        {
            std::stringstream nameTrame;
            if(i<10)
            {
                nameTrame << filePathName << "_000" << i << ".jpeg";
            }
            else if(i<100)
            {
                nameTrame << filePathName << "_00" << i << ".jpeg";
            }
            else
            {
                nameTrame << filePathName << "_0" << i << ".jpeg";
            }

            std::cout<<nameTrame.str()<<std::endl;

            sequence[i] = cv::imread(nameTrame.str());
        }

    }

    else if(format == VIDEO)
    {
        cv::VideoCapture capture;
        cv::Mat frame;

        capture.open(filePathName);

        if(capture.isOpened())
        {
            fps = capture.get(CV_CAP_PROP_FPS);

            while(true)
            {
                if(!capture.read(frame))
                    break;

                sequence.push_back(frame);
                frame.release();
            }

            capture.release();
            nbTrames = sequence.size();
        }

        else
        {
            std::cout<<"Impossible de lire la video : "<<filePathName<<std::endl;
        }
    }

    else
    {
        std::cout<<"Argument incorrect, veuillez entrer 'image' ou 'video'"<<std::endl;
    }
}



//------------------METHODES-HOG-------------------------------------------//


std::vector<cv::Rect> hogDetection(cv::Mat sequence, cv::HOGDescriptor hog)
{
    std::vector<cv::Rect> found;

    hog.detectMultiScale(sequence, found, 0, cv::Size(8,8), cv::Size(32,32), 1.05, 2);

    return found;
}



//------------------METHODES-BACKGROUND-SUBSTRACTION-----------------------//


void findUpperBody(cv::CascadeClassifier classifier, cv::Mat sequence, std::vector<cv::Rect> roi, std::vector<std::vector<cv::Rect>> &upper_body)
{
    cv::Mat sequenceGray;

    upper_body.clear();
    upper_body.resize(roi.size());

    cv::cvtColor(sequence, sequenceGray, CV_BGR2GRAY);
    cv::equalizeHist(sequenceGray, sequenceGray);

    for(size_t i=0;i<roi.size();i++)
    {
//        cv::imshow("temp", sequenceGray(roi[i]));
//        cv::waitKey(0);

        classifier.detectMultiScale(sequenceGray(roi[i]), upper_body[i], 1.1, 0, 0|CV_HAAR_SCALE_IMAGE, cv::Size(20, 20));

        for(size_t j=0;j<upper_body[i].size();j++)
        {
            upper_body[i][j].x += roi[i].x;
            upper_body[i][j].y += roi[i].y;
        }
    }
}


cv::Point rectCenter(cv::Rect rect)
{
    cv::Point center;

    center.x = rect.x + rect.width/2;
    center.y = rect.y + rect.height/2;

    return center;
}


void backgroundSubstractionDetection(cv::Mat sequence, std::vector<cv::Rect> &detectedPedestrianFiltered, cv::Ptr<cv::BackgroundSubtractor> &pMOG2, trackingOption &tracking)
{
    int threshold = 150;
    cv::Mat mask;
    cv::Mat sequenceGrayDiff;
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    std::vector<std::vector<cv::Point> > contours_poly;
    std::vector<cv::Rect> detectedPedestrian;

    pMOG2->apply(sequence,sequenceGrayDiff);


    cv::threshold(sequenceGrayDiff, mask, threshold, 255, cv::THRESH_BINARY);

    cv::erode(mask, mask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(6,6)));
    cv::dilate(mask, mask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(25,55)));
    cv::erode(mask, mask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,6)));


    /*
    cv::Mat dist;
    cv::distanceTransform(mask, dist, CV_DIST_L2, 3);
    cv::normalize(dist, dist, 0, 1., cv::NORM_MINMAX);
    cv::threshold(dist, dist, .4, 1., CV_THRESH_BINARY);
    cv::imshow("temp", dist);
    */

    cv::findContours(mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point(0,0));

    contours_poly.resize(contours.size());
    detectedPedestrian.resize(contours.size());

    for(size_t j=0;j<contours.size();j++)
    {
        cv::approxPolyDP(cv::Mat(contours[j]), contours_poly[j], 3, true);
        detectedPedestrian[j] = cv::boundingRect(cv::Mat(contours_poly[j]));


        //test
        /*
        double pix = 30;
        if(detectedPedestrian[j].x >= pix)
            detectedPedestrian[j].x -= pix;
        else
            detectedPedestrian[j].x = 0;
        if((detectedPedestrian[j].x+detectedPedestrian[j].width) <= (sequence.cols-pix))
            detectedPedestrian[j].width += pix;
        else
            detectedPedestrian[j].width = sequence.cols - detectedPedestrian[j].x;
        if(detectedPedestrian[j].y >= pix)
            detectedPedestrian[j].y -= pix;
        else
            detectedPedestrian[j].y = 0;
        if((detectedPedestrian[j].y+detectedPedestrian[j].height) <= (sequence.rows-pix))
            detectedPedestrian[j].height += pix;
        else
            detectedPedestrian[j].height = sequence.rows - detectedPedestrian[j].y;
        */
    }

    if(detectedPedestrian.size() != 0)
    {
        tracking = GOOD_FEATURES_TO_TRACK;
        detectedPedestrianFiltered.clear();
        detectedPedestrianFiltered.resize(detectedPedestrian.size());
        detectedPedestrianFiltered = detectedPedestrian;
    }
    else
        tracking = NOTHING_TO_TRACK;

}



//------------------METHODES-CORNERS-TRACKING------------------------//


std::vector<std::vector<cv::Point2f>> featuresDetection(cv::Mat sequence, std::vector<cv::Rect> found_filtered)
{
    cv::Mat sequenceGray;
    std::vector<std::vector<cv::Point2f>> corners;
    int maxCorners = 25;
    double qualityLevel = 0.01;
    double minDistance = 1.;
    int blockSize = 3;
    bool useHarrisDetector = false;
    double kdef = 0.04;

    cv::cvtColor(sequence, sequenceGray, CV_BGR2GRAY);

    corners.resize(found_filtered.size());

    for(size_t j=0;j<found_filtered.size();j++)
    {
        cv::goodFeaturesToTrack(sequenceGray(found_filtered[j]), corners[j], maxCorners, qualityLevel, minDistance, cv::noArray(), blockSize, useHarrisDetector, kdef);

        for(size_t k=0;k<corners[j].size();k++)
        {
            corners[j][k].x += found_filtered[j].x;
            corners[j][k].y += found_filtered[j].y;
        }
    }
    return corners;
}


std::vector<std::vector<cv::Point2f>> lucasKanadeTracking(cv::Mat previousSequence, cv::Mat sequence, std::vector<std::vector<cv::Point2f>> corners)
{
    cv::Mat previousSequenceGray;
    cv::Mat sequenceGray;
    std::vector<std::vector<cv::Point2f>> newCorners;
    std::vector<uchar> status;
    std::vector<float> err;

    cv::cvtColor(previousSequence, previousSequenceGray, CV_BGR2GRAY);
    cv::cvtColor(sequence, sequenceGray, CV_BGR2GRAY);

    newCorners.resize(corners.size());

    for(size_t j=0;j<corners.size();j++)
    {
        cv::calcOpticalFlowPyrLK(previousSequenceGray, sequenceGray, corners[j], newCorners[j], status, err);
    }
    return newCorners;
}



//------------------METHODES-OPTICAL-FLOW-FARNEBACK------------------------//


void drawOptFlowMap(const cv::Mat& flow, cv::Mat& cflowmap, int step, const cv::Scalar& color)
{
        for(int y = 0; y < cflowmap.rows; y += step)
            for(int x = 0; x < cflowmap.cols; x += step)
            {
                const cv::Point2f& fxy = flow.at<cv::Point2f>(y, x);
                cv::line(cflowmap, cv::Point(x,y), cv::Point(cvRound(x+fxy.x), cvRound(y+fxy.y)), color);
                cv::circle(cflowmap, cv::Point(x,y), 2, color, -1);
            }
}



//------------------METHODES-CAMSHIFT-ET-KALMAN-FILTER---------------------//


std::vector<cv::MatND> computeProbImage(cv::Mat image, std::vector<cv::Rect> rectRoi, std::vector<cv::Mat> &hist, std::vector<bool> &detected)
{
    int smin = 30;
    int vmin = 10;
    int vmax = 256;
    cv::Mat mask;
    cv::Mat hsv;
    cv::Mat hue;
    std::vector<cv::MatND> backProj;
    int channels[] = {0,0};
    int hbins = 30;                                   // Quantize the hue to 30 levels
    //int sbins = 32;                                 
    int histSize = MAX( hbins, 2 );
    //int histSizes[] = {hbins, sbins};
    float hue_range[] = { 0, 180 };                   // hue varies from 0 to 179, see cvtColor
    //float sat_range[] = { 0, 256 };                 
    const float* range = { hue_range };               
    //const float* ranges = { hue_range, sat_range };
    //double maxVal=0;

    backProj.resize(rectRoi.size());
    hist.resize(rectRoi.size());

    cv::cvtColor(image, hsv, CV_BGR2HSV);
    hue.create(hsv.size(), hsv.depth());
    cv::mixChannels(&hsv, 1, &hue, 1, channels, 1);
    cv::inRange(hsv, cv::Scalar(0, smin, MIN(vmin,vmax)), cv::Scalar(180, 256, MAX(vmin, vmax)), mask);

    for(size_t i=0;i<rectRoi.size();i++)
    {
        if(!detected[i])
        {
            cv::Mat roi(hue, rectRoi[i]);
            cv::Mat maskroi(mask, rectRoi[i]);

            cv::calcHist(&roi, 1, 0, maskroi, hist[i], 1, &histSize, &range, true, false);
            cv::normalize(hist[i], hist[i], 0, 255, cv::NORM_MINMAX);

            detected[i] = true;

            roi.release();
            maskroi.release();
        }

        cv::calcBackProject(&hue, 1, 0, hist[i], backProj[i], &range);
        backProj[i] &= mask;
    }

    return backProj;
}



//------------------METHODES-DETECTION ROI-FOR-KALMAN-------------------//

cv::Rect findROI(cv::Point predict, std::vector<cv::Rect> pedestrian)
{
    cv::Rect roi;
    double dist = 0;
    double distmin = 99999;

    for(size_t i=0;i<pedestrian.size();i++)
    {
        dist = cv::norm(cv::Mat(predict), cv::Mat(rectCenter(pedestrian[i])));

        if(dist < distmin)
        {
            roi = pedestrian[i];
            distmin = dist;
        }
    }

    return roi;
}



int main(int argc, char *argv[])
{
    choiceAlgo algo;
    formatVideo format;
    std::string inputFileName(argv[3]);
    int nbTrames = 501;
    double fps = 0;

    std::vector<cv::Mat> sequence;

    cv::Mat previousSequence;

    int nbPedestrians = 0;

    cv::HOGDescriptor hog;
    hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());

    std::vector<cv::Rect> detectedPedestrian;


    // HOG + Good feature to track + LK
    std::vector<std::vector<cv::Point2f>> featuresDetected;
    std::vector<std::vector<cv::Point2f>> previousFeaturesDetected;



    // HOG + Template tracking
    std::vector<cv::Rect> boxes;
    std::vector<cv::Rect> previousBoxes;



    // Optical flow farneback
    cv::Mat flow;
    cv::Mat imGray;
    cv::Mat imGrayPrev;



    //camshift and kalman filter
    std::vector<cv::MatND> backProj;
    std::vector<cv::Rect> roiHogDetected;
    std::vector<cv::Rect> roiCamShift;
    std::vector<bool> detected;
    std::vector<cv::Mat> hist;
    std::vector<cv::RotatedRect> rectCamShift;
    cv::Point2f rect_points[4];

    trackingOption tracking;
    cv::Ptr<cv::BackgroundSubtractor> pMOG2;
    std::vector<cv::Rect> detectedPedestrianFiltered;

    cv::KalmanFilter KF(4,2,0,CV_32F);
    cv::Mat_<float> measurement(2,1);
    cv::Mat prediction;
    cv::Mat estimated;

    pMOG2 = cv::createBackgroundSubtractorMOG2();

	// 하르 케스케이드
    std::vector<std::vector<cv::Rect>> rect_upper_body;
    cv::CascadeClassifier classifier;
    std::string upper_body_cascade_name = "haarcascade_fullbody.xml";

    if(!classifier.load(upper_body_cascade_name))
    {
        std::cout<<"le fichier "<<upper_body_cascade_name<<" ne peut etre charge"<<std::endl;
        return -1;
    }

    //--------------------------------------------------------------------------------------//


    //Background substraction and Kalman
    bool initKalman = true;
    cv::KalmanFilter Kalman(4,2,0,CV_32F);
    cv::Mat_<float> measurmt(2,1);
    cv::Mat predict;
    cv::Mat estim;

    Kalman.transitionMatrix.at<float>(0,0) = 1;
    Kalman.transitionMatrix.at<float>(0,1) = 0;
    Kalman.transitionMatrix.at<float>(0,2) = 1;
    Kalman.transitionMatrix.at<float>(0,3) = 0;
    Kalman.transitionMatrix.at<float>(1,0) = 0;
    Kalman.transitionMatrix.at<float>(1,1) = 1;
    Kalman.transitionMatrix.at<float>(1,2) = 0;
    Kalman.transitionMatrix.at<float>(1,3) = 1;
    Kalman.transitionMatrix.at<float>(2,0) = 0;
    Kalman.transitionMatrix.at<float>(2,1) = 0;
    Kalman.transitionMatrix.at<float>(2,2) = 1;
    Kalman.transitionMatrix.at<float>(2,3) = 0;
    Kalman.transitionMatrix.at<float>(3,0) = 0;
    Kalman.transitionMatrix.at<float>(3,1) = 0;
    Kalman.transitionMatrix.at<float>(3,2) = 0;
    Kalman.transitionMatrix.at<float>(3,3) = 1;

    measurmt.setTo(cv::Scalar(0));

    cv::setIdentity(Kalman.measurementMatrix);
    cv::setIdentity(Kalman.processNoiseCov, cv::Scalar::all(1e-4));
    cv::setIdentity(Kalman.measurementNoiseCov, cv::Scalar::all(1e-1));
    cv::setIdentity(Kalman.errorCovPost, cv::Scalar::all(.1));


    cv::Rect rectK;
    cv::Rect prevRectK;
    algo = detectAlgo(std::string(argv[1]));
    format = detectFormat(std::string(argv[2]));





    //------------------VIDEO--------------------------------------------------//

    if(format == SEQUENCE_IMAGE)
        sequence.resize(nbTrames);
    else if(format == VIDEO)
        std::cout<<"video"<<std::endl;

    extractVideoData(sequence, format, inputFileName, nbTrames, fps);

    cv::namedWindow("Video", cv::WINDOW_AUTOSIZE);



    //------------------TRAITEMENT-VIDEO---------------------------------------//

    for(int i=0;i<nbTrames;i++)
    {

        if(i>0)
            previousSequence = sequence[i-1];
        else
            previousSequence = sequence[i];




        ///------------------HOG + Good Features to track + LK-----------------//

        if(algo == HOG_GOODFEATURESTOTRACK_LK)
        {
            if(i%20 == 0)
            {
                detectedPedestrian = hogDetection(sequence[i], hog);
                nbPedestrians = detectedPedestrian.size();

                if(nbPedestrians != 0)
                {
                    featuresDetected = featuresDetection(sequence[i], detectedPedestrian);
                    previousFeaturesDetected.resize(featuresDetected.size());
                    previousFeaturesDetected = featuresDetected;
                }
            }
            else if(previousFeaturesDetected.size() != 0)
            {
                featuresDetected = lucasKanadeTracking(previousSequence, sequence[i], previousFeaturesDetected);

                previousFeaturesDetected.clear();
                previousFeaturesDetected.resize(featuresDetected.size());
                previousFeaturesDetected = featuresDetected;
            }


            //--------Representation--------------------

            /*
            cv::Scalar myColor;

            for(size_t j=0;j<featuresDetected.size();j++)
            {
                if(j%3 == 0)
                    myColor = cv::Scalar(0,0,cv::RNG().uniform(200,255));

                else if(j%2 == 0)
                    myColor = cv::Scalar(0,cv::RNG().uniform(200,255),0);

                else
                    myColor = cv::Scalar(cv::RNG().uniform(200,255),0,0);

                for(size_t k=0;k<featuresDetected[j].size();k++)
                {
                    cv::circle(sequence[i], featuresDetected[j][k], 1, myColor,-1);
                }
            }
            */


            for(size_t j=0;j<featuresDetected.size();j++)
            {
                cv::rectangle(sequence[i], cv::boundingRect(featuresDetected[j]), cv::Scalar( 0, 0, 255), 2, 8, 0 );
            }

            cv::imshow("Video", sequence[i]);
        }





        ///------------------HOG + Template Tracking---------------------------//

        else if(algo == HOG_TEMPLATE_TRACKING)
        {
            if(i%20 == 0)
            {
                detectedPedestrian = hogDetection(sequence[i], hog);
                nbPedestrians = detectedPedestrian.size();

                if(nbPedestrians != 0)
                {
                    boxes = templateTracking(sequence[i], detectedPedestrian, CV_TM_CCORR_NORMED);
                    previousBoxes.resize(boxes.size());
                    previousBoxes = boxes;
                }
            }
            else if(previousBoxes.size() != 0)
            {
                boxes = templateTracking(sequence[i], previousBoxes, CV_TM_CCORR_NORMED);

                previousBoxes.clear();
                previousBoxes.resize(boxes.size());
                previousBoxes = boxes;
            }

            //--------Representation--------------------

            for(size_t j=0;j<boxes.size();j++)
            {
                cv::rectangle(sequence[i], boxes[j], cv::Scalar( 0, 0, 255), 2, 8, 0 );
            }
            cv::imshow("Video", sequence[i]);
        }






        ///------------------HOG + Optical Flow Farneback----------------------//

        else if(algo == OPT_FLOW_FARNEBACK)
        {
            if(i!=0)
            {
                flow = cv::Mat::zeros(sequence[i].size(), CV_32FC2);
                cv::cvtColor(sequence[i], imGray, CV_BGR2GRAY);
                cv::cvtColor(sequence[i-1], imGrayPrev, CV_BGR2GRAY);

                cv::calcOpticalFlowFarneback(imGrayPrev, imGray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);


                //-----------------Representation------------------------------//

                drawOptFlowMap(flow, imGrayPrev, 16, CV_RGB(0, 255, 0)); //dessin test
                cv::imshow("Video", imGrayPrev);
            }
        }





        ///--------------HOG+Camshift + Kalman Filter--------------------------//

        else if(algo == CAMSHIFT_KALMAN_FILTER)
        {

            //camshift
            if(i%20 == 0 && roiCamShift.size() == 0)
            {
                roiHogDetected = hogDetection(sequence[i], hog);
                refineROI(roiCamShift, detected, roiHogDetected);

                //test
                if(roiCamShift.size() != 0)
                {
                    /*
                    roiCamShift[0].x += 30;
                    roiCamShift[0].width -= 60;
                    roiCamShift[0].y += 40;
                    roiCamShift[0].height -= 100;
                    */
                    roiCamShift[0].x += roiCamShift[0].width/2;
                    roiCamShift[0].width = roiCamShift[0].width/3;
                    //roiCamShift[0].y += roiCamShift[0].height/2;
                    roiCamShift[0].height = roiCamShift[0].height/3;
                }
                
            }

            backProj = computeProbImage(sequence[i], roiCamShift, hist, detected);

            if (roiCamShift.size() != 0)
                cv::imshow("temp", backProj[0]);

            rectCamShift.resize(roiCamShift.size());

            for(size_t j=0;j<roiCamShift.size();j++)
            {
                /*
                std::cout<<roiCamShift[j]<<std::endl;
                cv::rectangle(backProj[j], roiCamShift[j], cv::Scalar( 255, 0, 0), 2, 8, 0 ); //DEBUG
                cv::imshow("before camshift", backProj[j]);
                cv::waitKey(0);
                */
                cv::Rect rectMeanShift;
                rectMeanShift = roiCamShift[j];
                cv::meanShift(backProj[j], rectMeanShift, cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 10, 1));

                cv::rectangle(sequence[i], rectMeanShift, cv::Scalar( 0, 255, 0), 2, 8, 0 );

                rectCamShift[j] = cv::CamShift(backProj[j], roiCamShift[j], cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 10, 1));
                rectCamShift[j].points(rect_points);


                for(int k = 0; k < 4; k++)
                    cv::line(sequence[i], rect_points[k], rect_points[(k+1)%4], cv::Scalar( 0, 0, 255), 2, 8);
            }

            for(size_t j=0;j<roiCamShift.size();j++)
                cv::rectangle(sequence[i], roiCamShift[j], cv::Scalar( 255, 0, 0), 2, 8, 0 );
            cv::imshow("Video", sequence[i]);
        }




		
        ///------------------BACKGROUND-SUBSTRACTION---------------------------//

        else if(algo == BACKGROUND_LK)
        {
            if(i%10 == 0)
            {
                backgroundSubstractionDetection(sequence[i], detectedPedestrianFiltered, pMOG2, tracking);
            }

            if(tracking == GOOD_FEATURES_TO_TRACK)
            {
                featuresDetected.resize(detectedPedestrianFiltered.size());
                featuresDetected = featuresDetection(sequence[i], detectedPedestrianFiltered);
                previousFeaturesDetected.resize(featuresDetected.size());
                previousFeaturesDetected = featuresDetected;

                tracking = LUCAS_KANADE;

                KF.transitionMatrix.at<float>(0,0) = 1;
                KF.transitionMatrix.at<float>(0,1) = 0;
                KF.transitionMatrix.at<float>(0,2) = 1;
                KF.transitionMatrix.at<float>(0,3) = 0;
                KF.transitionMatrix.at<float>(1,0) = 0;
                KF.transitionMatrix.at<float>(1,1) = 1;
                KF.transitionMatrix.at<float>(1,2) = 0;
                KF.transitionMatrix.at<float>(1,3) = 1;
                KF.transitionMatrix.at<float>(2,0) = 0;
                KF.transitionMatrix.at<float>(2,1) = 0;
                KF.transitionMatrix.at<float>(2,2) = 1;
                KF.transitionMatrix.at<float>(2,3) = 0;
                KF.transitionMatrix.at<float>(3,0) = 0;
                KF.transitionMatrix.at<float>(3,1) = 0;
                KF.transitionMatrix.at<float>(3,2) = 0;
                KF.transitionMatrix.at<float>(3,3) = 1;

                measurement.setTo(cv::Scalar(0));

                for(size_t j=0;j<featuresDetected.size();j++)
                {
                    detectedPedestrianFiltered[j] = cv::boundingRect(featuresDetected[j]);
                }

                KF.statePre.at<float>(0) = rectCenter(detectedPedestrianFiltered[0]).x;
                KF.statePre.at<float>(1) = rectCenter(detectedPedestrianFiltered[0]).y;
                KF.statePre.at<float>(2) = 0;
                KF.statePre.at<float>(3) = 0;

                cv::setIdentity(KF.measurementMatrix);
                cv::setIdentity(KF.processNoiseCov, cv::Scalar::all(1e-4));
                cv::setIdentity(KF.measurementNoiseCov, cv::Scalar::all(1e-1));
                cv::setIdentity(KF.errorCovPost, cv::Scalar::all(.1));
            }

            else if(tracking == LUCAS_KANADE)
            {
                for(size_t j=0;j<featuresDetected.size();j++)
                {
                    detectedPedestrianFiltered[j] = cv::boundingRect(featuresDetected[j]);
                }

                featuresDetected = lucasKanadeTracking(previousSequence, sequence[i], previousFeaturesDetected);

                previousFeaturesDetected.clear();
                previousFeaturesDetected.resize(featuresDetected.size());
                previousFeaturesDetected = featuresDetected;

                prediction = KF.predict();
                cv::Point predictPt(prediction.at<float>(0),prediction.at<float>(1));

                // Get mouse point
                measurement(0) = rectCenter(detectedPedestrianFiltered[0]).x;
                measurement(1) = rectCenter(detectedPedestrianFiltered[0]).y;

                cv::Point measPt(measurement(0),measurement(1));

                // The "correct" phase that is going to use the predicted value and our measurement
                cv::Mat estimated = KF.correct(measurement);
                cv::Point statePt(estimated.at<float>(0),estimated.at<float>(1));

                //cv::circle(sequence[i], measPt, 1, cv::Scalar(0,255,0), 7, 24);
                //cv::circle(sequence[i], predictPt, 1, cv::Scalar(0,255,255), 7, 24);
            }


            //--------Representation--------------------

            if(tracking != NOTHING_TO_TRACK)
                findUpperBody(classifier, sequence[i], detectedPedestrianFiltered, rect_upper_body);

            for(size_t j=0;j<featuresDetected.size();j++)
            {

                for(size_t k=0;k<rect_upper_body[j].size();k++)
                {
                    cv::rectangle(sequence[i], rect_upper_body[j][k], cv::Scalar( 0, 255, 0), 2, 8, 0 );
                }
                //detectedPedestrianFiltered[j] = cv::boundingRect(featuresDetected[j]);
                cv::rectangle(sequence[i], cv::boundingRect(featuresDetected[j]), cv::Scalar( 0, 0, 255), 2, 8, 0 );
            }


            //affichage de la video
            cv::imshow("Video", sequence[i]);
        }


        detectedPedestrian.clear();
        featuresDetected.clear();
        boxes.clear();

        previousSequence.release();

        flow.release();
        imGray.release();
        imGrayPrev.release();

        roiHogDetected.clear();
        backProj.clear();

        //------------------CONDITIONS-ARRET-----------------------------------//

        if (cv::waitKey((int)(1000/fps)) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
        {
            std::cout << "esc key is pressed by user" << std::endl;
            return 0;
        }
    }

    cv::waitKey(0);
    return 0;
}
