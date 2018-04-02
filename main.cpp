#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/video.hpp"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <stdio.h>
#include <time.h>
#include <vector>
#include "findmissing.h"
#include <armadillo>
#include "kalman.hpp"

/*
open and play video: http://opencv-srf.blogspot.ca/2011/09/capturing-images-videos.html
ForegroundMaskMOG2: http://docs.opencv.org/3.2.0/d1/dc5/tutorial_background_subtraction.html
*/

using namespace cv;
using namespace std;
//using namespace arma;

cv::Mat frame;
cv::Mat fgMaskMOG2;
Ptr<BackgroundSubtractor> pMOG2;
int thres = 180;
cv::Mat mog2frame;
cv::Mat outputmog2;
cv::Mat framemog2;

int trackidcount = 1;
int mindis = 0;

int main(void)
{
    VideoCapture cap("D:/Dropbox/McMaster/ECE769 Multimodel Video Tracking and Fusion/assignment2/at2.avi");

    if ( !cap.isOpened() )  // if not success, exit program
    {
         cout << "Cannot open the video file" << endl;
         return -1;
    }
    double hsize = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    double wsize = cap.get(CV_CAP_PROP_FRAME_WIDTH);
    double fps = cap.get(CV_CAP_PROP_FPS); //get the frames per seconds of the video
    cout << "Fps of Original Video: " << fps << endl;
    pMOG2 = createBackgroundSubtractorMOG2();

    cv::Mat initbg(hsize,wsize,CV_8UC1,Scalar(0));
    initbg.copyTo(mog2frame);

    int erosize = 2;
    cv::Mat element = getStructuringElement(MORPH_RECT,Size(2*erosize + 1,2*erosize+1),Point(erosize,erosize));
    int dilsize = 10;
    cv::Mat dilelement = getStructuringElement(MORPH_ELLIPSE,Size(2*dilsize + 1,2*dilsize+1),Point(dilsize,dilsize));

    std::vector<vector<Point> > contours;

    bool pauseplay = true;

    int framecount = 1;
    float fetchinit;
    float fetchtimeb;
    float fetchtime;

    int maxdetect = 100;
    int infocount = 5;
    vector<double> info(maxdetect*infocount);


    vector<int> forclearing(maxdetect);
    vector<int> forclearing2(maxdetect*infocount);
    vector<int> trackid(maxdetect);
    vector<int> newtrackid(maxdetect);
    vector<int> asstrackid(maxdetect);
    vector<int> unasstrackid(maxdetect);
    vector<int> ptargetx(maxdetect*infocount);
    vector<int> ptargety(maxdetect*infocount);
    vector<int> ptargetw(maxdetect*infocount);
    vector<int> ptargeth(maxdetect*infocount);
    vector<int> ctargetx(maxdetect*infocount);
    vector<int> ctargety(maxdetect*infocount);
    vector<int> ctargetw(maxdetect*infocount);
    vector<int> ctargeth(maxdetect*infocount);
    vector<int> dist(maxdetect*maxdetect);
    vector<int> disttemp(maxdetect);

    vector<int> asstargetx(maxdetect*infocount);
    vector<int> unasstargetx(maxdetect*infocount);
    vector<int> asstargety(maxdetect*infocount);
    vector<int> unasstargety(maxdetect*infocount);
    vector<int> asstargetw(maxdetect*infocount);
    vector<int> unasstargetw(maxdetect*infocount);
    vector<int> asstargeth(maxdetect*infocount);
    vector<int> unasstargeth(maxdetect*infocount);

    vector<int> sameat(maxdetect);

//    vector<int> trackcount(maxdetect);
//    vector<float> trackstats(maxdetect);
//    vector<int> tentrackid(maxdetect);
//    vector<int> comtrackid(maxdetect);
//    vector<int> termtrackid(maxdetect);
//    vector<int> trackhist(maxdetect*maxdetect);
    arma::mat mattrackid(1,maxdetect);
    arma::mat matctargetx(1,maxdetect);
    arma::mat matctargety(1,maxdetect);
    arma::mat matctargetw(1,maxdetect);
    arma::mat matctargeth(1,maxdetect);
    arma::mat trackhist(500,maxdetect);
    arma::mat targetxhist(500,maxdetect);
    arma::mat targetyhist(500,maxdetect);
    arma::mat targetwhist(500,maxdetect);
    arma::mat targethhist(500,maxdetect);
//    arma::mat tracktest(5,maxdetect);
    arma::mat termtrackid(1,maxdetect);
    arma::mat comtrackid(1,maxdetect);
    arma::mat tentrackid(1,maxdetect);
    arma::mat trackcount(1,maxdetect);
    arma::mat tentrackcount(1,maxdetect);
    arma::mat comtarget(2,maxdetect);
    arma::mat comhist(500,maxdetect);
    arma::mat comtrackcount(1,maxdetect);
    arma::mat nocomtrackhist(2,500);
    arma::mat termtrackhist(2,500);

    arma::mat pxy(4,4*maxdetect);
    arma::mat pwh(4,4*maxdetect);
    arma::mat xxy(4,4*maxdetect);
    arma::mat xwh(4,4*maxdetect);
    float R=16;
    float inp=25;
    arma::mat px(maxdetect,500);
    arma::mat cx(maxdetect,500);
    arma::mat py(maxdetect,500);
    arma::mat cy(maxdetect,500);
    arma::mat pw(maxdetect,500);
    arma::mat cw(maxdetect,500);
    arma::mat ph(maxdetect,500);
    arma::mat ch(maxdetect,500);

    arma::mat pxyt(4,4);
    arma::mat pwht(4,4);
    arma::mat xxyt(4,4);
    arma::mat xwht(4,4);

    while(1)
    {
        cv::Mat frame;
        bool bSuccess = cap.read(frame);

        if (!bSuccess) //if not success, break loop
        {
            cout << "Cannot read the frame from video file" << endl;
            abort();
        }
        //imshow("Original Video", frame);

        if(!cap.read(frame)) {
            cerr << "Unable to read next frame." << endl;
            cerr << "Exiting..." << endl;
            exit(EXIT_FAILURE);
        }
        frame.copyTo(framemog2);
        pMOG2->apply(frame, fgMaskMOG2);

        unsigned char *mog2 = (unsigned char*)(fgMaskMOG2.data);

        for (int i=1; i<=frame.rows; i++){
            for (int j=1; j<=frame.cols; j++){
                unsigned char x = mog2[fgMaskMOG2.step * i + j];
                unsigned char xx;
                if (x > thres){
                    xx = (int)255;
                }
                if (x < thres){
                    xx = (int)0;
                }
                mog2frame.at<unsigned char>(i,j) = xx;
            }
        }
        //imshow("MOG2",fgMaskMOG2);
        //imshow("fgMaskMOG2", mog2frame);

        mog2frame.copyTo(outputmog2);
        erode(outputmog2,outputmog2,element);
        dilate(outputmog2,outputmog2,dilelement);
        erode(outputmog2,outputmog2,element);
        erode(outputmog2,outputmog2,element);

//        imshow("outputmog2",outputmog2);
//        int firstframeid=0;
        findContours(outputmog2, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
        std::vector<std::vector<Point> >::const_iterator bcount = contours.begin();
        std::vector<std::vector<Point> >::const_iterator end = contours.end();
        int detectcount = 0;
        int minwh = 10;
        cout << "-------------------------------------------------------" << endl;

        while (bcount != end) {
            Rect bounds = boundingRect(*bcount);
            if(framecount>1){
                minwh=20;
            }
            if((bounds.width > minwh) && (bounds.height > minwh)){
                rectangle(framemog2, bounds, cv::Scalar(0,255,0),2);
                //cout << bounds.width << ", " << bounds.height << ", " << bounds.x << ", " << bounds.y << endl;

                info[detectcount*infocount + 0] = (int)detectcount;
                info[detectcount*infocount + 1] = (int)bounds.x;
                info[detectcount*infocount + 2] = (int)bounds.y;
                info[detectcount*infocount + 3] = (int)bounds.width;
                info[detectcount*infocount + 4] = (int)bounds.height;
                detectcount++;
            }
            ++bcount;
//            rectangle(framemog2, bounds, cv::Scalar(0,255,0),1);
//            //cout << bounds.width << ", " << bounds.height << ", " << bounds.x << ", " << bounds.y << endl;
//            ++bcount;
//            info[detectcount*infocount + 0] = (int)detectcount;
//            info[detectcount*infocount + 1] = (int)bounds.x;
//            info[detectcount*infocount + 2] = (int)bounds.y;
//            info[detectcount*infocount + 3] = (int)bounds.width;
//            info[detectcount*infocount + 4] = (int)bounds.height;
//            detectcount++;



            //print id
//            if(framecount <= 100){
//                char idname[maxdetect];
//                sprintf(idname,"ID: %i",firstframeid);
//                Point idplace(bounds.x,bounds.y);
//                putText(framemog2,idname,idplace,FONT_HERSHEY_SIMPLEX,0.5,Scalar(0,0,255),2);
//            }
//            firstframeid++;
        }
        char framename[1];
        sprintf(framename,"Frame: %i",framecount);
        Point txtorg(0,30);
        putText(framemog2, framename,txtorg,FONT_HERSHEY_SIMPLEX,1,Scalar(0,0,255),2);
        imshow("OutputMOG2",framemog2);


        cout << "Frame: " << framecount << endl;
        cout << "Detections: " << detectcount << endl;
        cout << "ID\t" << "x\t" << "y\t" << "width\t" << "height" << endl;
        cout << "---------------------------------------" << endl;
        int autocount = 0;
        for(auto vec : info)
        {
            if(autocount < (detectcount*infocount)){
            cout<<vec<<"\t";
                if((autocount+1) % 5 == 0){
                    cout << endl;
                }
            autocount++;
            }
        }
        cout << "---------------------------------------" << endl;

        if (framecount == 1){
            fetchinit = clock();
            fetchtimeb = fetchinit;

        }
        if (framecount > 1){
            fetchtime = clock()-fetchtimeb;
            fetchtimeb = clock();
            fetchtime = (fetchtime/CLOCKS_PER_SEC);
        }

        cout << "T: " <<fetchtime << endl;
        float T = fetchtime;








        //Tracking Starts here
        //data associate
        int pastdetectcount;
        vector<int> minvalues(maxdetect);   //per track
        vector<int> measuren(maxdetect);    //associate min measurement number for each track
        vector<int> measuren2(maxdetect);   // min track number for each measrement
        vector<int> trackn(maxdetect);      //number of tracks
        sameat.swap(forclearing);


        if(framecount==1){
            cout << "Initialize Tracks: ";
            for(int k=0; k<detectcount; k++){
                trackid[k]=k+1;
                ptargetx[k]=info[(k)*infocount + 1];
                ptargety[k]=info[(k)*infocount + 2];

                ptargetw[k]=info[(k)*infocount + 3];
                ptargeth[k]=info[(k)*infocount + 4];
                //cout << trackid[k-1] << endl;
//                cout <<targetx[k]<< endl;
//                cout <<targety[k]<<endl;

                tentrackid(0,k)=trackid[k];
//                trackhist[(framecount-1)*maxdetect+k]=trackid[k];
                trackcount[k]=1;
                cout << trackidcount << " ; ";
                trackidcount++;
                px(0,k)=ptargetx[k];
                py(0,k)=ptargety[k];
                pw(0,k)=ptargetw[k];
                ph(0,k)=ptargeth[k];

                pxy(arma::span(0,3),arma::span(k*4,k*4+3))=25*arma::eye(4,4);
                pwh(arma::span(0,3),arma::span(k*4,k*4+3))=25*arma::eye(4,4);
//                pxy(arma::span(0,3),arma::span(k*4,k*4+3))=25*arma::eye(4,4);

                xxy(0,k)=px(0,k);
                xxy(2,k+2)=py(0,k);
                xwh(0,k)=pw(0,k);
                xwh(2,k+2)=ph(0,k);

            }
            cout << endl;
//            cout << pxy<<endl;
            pastdetectcount=detectcount;
            comtrackid.fill(0);
        }
        else{

//                ctargetx.clear();
//                ctargety.clear();
//                ctargetw.clear();
//                ctargeth.clear();
                 for(int k=0; k<detectcount; k++){
                    ctargetx[k]=info[(k)*infocount + 1];
                    ctargety[k]=info[(k)*infocount + 2];
                    ctargetw[k]=info[(k)*infocount + 3];
                    ctargeth[k]=info[(k)*infocount + 4];

                    cx(k,framecount-1)=ctargetx[k];
                    cy(k,framecount-1)=ctargety[k];
                    ch(k,framecount-1)=ctargeth[k];
                    cw(k,framecount-1)=ctargetw[k];


                    xxy(0,k)=ctargetx[k];
                    xxy(2,k+2)=ctargety[k];
                    xwh(0,k)=ctargetw[k];
                    xwh(2,k+2)=ctargeth[k];
//                    trackidcount++;
//                    newtrackid[k]=trackidcount;
                    }
                if(pastdetectcount>detectcount){
                    for(int k=0; k<detectcount; k++){
                        for(int i=0; i<pastdetectcount; i++){
                            dist[k*detectcount+i]=pow((ctargetx[k]-ptargetx[i]),2)+pow((ctargety[k]-ptargety[i]),2);
                            disttemp[i]=dist[k*detectcount+i];
                        }
                        int mindis=disttemp[0];
                        int closest=0;
                        for(int i=1; i<pastdetectcount; i++){
                            if(disttemp[i]<mindis){
                                mindis=disttemp[i];
                                closest=i;
                            }
                        }
                        trackn[k]=k;
                        measuren2[k]=closest;
                        minvalues[k]=mindis;
//                        cout << k << endl;
//                        cout << closest << endl;
                    }

                }
                if(pastdetectcount<detectcount){
                     for(int k=0; k<pastdetectcount; k++){
                        for(int i=0; i<detectcount; i++){

                                dist[k*pastdetectcount+i]=pow((ctargetx[i]-ptargetx[k]),2)+pow((ctargety[i]-ptargety[k]),2);
    //                            cout << dist[k*pastdetectcount+i] << endl;
                                disttemp[i]=dist[k*pastdetectcount+i];
//                                cout << "IIIIIIIIII"<< i <<endl;
//                                cout << "cx" << ctargetx[i] << "cy" << ctargety[i];
//                                cout <<" px" << ptargetx[k] <<" py" << ptargety[k];
//                                cout <<"   distemp"<<disttemp[i] << "    " ;
                        }
//                        cout << endl;
                        int mindis=disttemp[0];
                        int closest=0;
                        for(int i=1; i<detectcount; i++){
                            if(disttemp[i]<mindis){
                                mindis=disttemp[i];
                                closest=i;
                            }
                        }
                        trackn[k]=k;
                        measuren[k]=closest;
                        minvalues[k]=mindis;
//                        cout <<"     " << trackn[k] << endl;
//                        cout <<"     " << measuren[k] << endl;
//                        cout << "     " <<minvalues[k] << endl;
                    }
                }
                if(pastdetectcount == detectcount){
                     for(int k=0; k<pastdetectcount; k++){
                        for(int i=0; i<detectcount; i++){

                                dist[k*pastdetectcount+i]=pow((ctargetx[i]-ptargetx[k]),2)+pow((ctargety[i]-ptargety[k]),2);
    //                            cout << dist[k*pastdetectcount+i] << endl;
                                disttemp[i]=dist[k*pastdetectcount+i];
//                                cout << "IIIIIIIIII"<< i <<endl;
//                                cout << "cx" << ctargetx[i] << "cy" << ctargety[i];
//                                cout <<" px" << ptargetx[k] <<" py" << ptargety[k];
//                                cout <<"   distemp"<<disttemp[i] << "    " ;
                        }
//                        cout << endl;
                        int mindis=disttemp[0];
                        int closest=0;
                        for(int i=1; i<detectcount; i++){
                            if(disttemp[i]<mindis){
                                mindis=disttemp[i];
                                closest=i;
                            }
                        }
                        trackn[k]=k;
                        measuren[k]=closest;
                        minvalues[k]=mindis;
//                        cout <<"     " << trackn[k] << endl;
//                        cout <<"     " << measuren[k] << endl;
//                        cout << "     " <<minvalues[k] << endl;
                    }
                }


//            ptargetx.clear();
//            ptargety.clear();
//            ptargetw.clear();
//            ptargeth.clear();
            ptargetx.swap(forclearing);
            ptargety.swap(forclearing);
            ptargetw.swap(forclearing);
            ptargeth.swap(forclearing);
            asstrackid.clear();
            asstargetx.clear();
            asstargety.clear();
            asstargetw.clear();
            asstargeth.clear();
            unasstrackid.clear();
            unasstargetx.clear();
            unasstargety.clear();
            unasstargetw.clear();
            unasstargeth.clear();
            newtrackid.clear();

            if(pastdetectcount==0){
                    for(int k=0; k<detectcount; k++){
                        unasstargetx[k]=ctargetx[k];
                        unasstargety[k]=ctargety[k];
                        unasstargetw[k]=ctargetw[k];
                        unasstargeth[k]=ctargeth[k];

                        trackidcount++;
                        newtrackid[k]=trackidcount;
                        cout << "Unassociated Targets: (" << unasstargetx[k] << "," << unasstargety[k] << ");";
                        cout << "Initialise new tracks: " << trackidcount << endl;
    //                    unassn=detectcount-pastdetectcount;
    //                    unassn++;
                        ptargetx[k]=unasstargetx[k];
                        ptargety[k]=unasstargety[k];
                        ptargetw[k]=unasstargetw[k];
                        ptargeth[k]=unasstargeth[k];
                        trackid[k]=newtrackid[k];


                        xxy(0,4*(k))=ptargetx[k];
                        xxy(2,4*(k)+2)=ptargety[k];
                        xwh(0,4*(k))=ptargetw[k];
                        xwh(2,4*(k)+2)=ptargeth[k];


                        arma::uvec mintentrack=find(tentrackid==0);
    //cout<<arma::min(mintentrack)<<endl;
                        tentrackid(0,arma::min(mintentrack))=newtrackid[k];
                    }

            }
            else{
            if(detectcount<pastdetectcount){
                int assn=0;
                int unassn=0;
                for(int i = 0; i<pastdetectcount; i++){
//                        cout<< i <<"   "<< measuren2[i]<< endl;
//                        cout << trackid[i]<< endl;
                        if(i<detectcount){
//                                cout<< i <<"   "<< measuren2[i]<< endl;
//                        cout << trackid[i]<< endl;
//                        cout <<trackid[measuren2[i]]<< endl;
                            asstrackid[assn]=trackid[measuren2[i]];
                            asstargetx[assn]=ctargetx[measuren2[i]];
                            asstargety[assn]=ctargety[measuren2[i]];
                            asstargetw[assn]=ctargetw[measuren2[i]];
                            asstargeth[assn]=ctargeth[measuren2[i]];
                            cout << "Target at (" << asstargetx[assn] << "," << asstargety[assn] << "); ";
                            cout << "Associate with Track ID: " << asstrackid[assn] << endl;


                            pxyt=pxy(arma::span(0,3),arma::span(measuren2[i]*4,measuren2[i]*4+3));
                            xxyt=xxy(arma::span(0,3),arma::span(measuren2[i]*4,measuren2[i]*4+3));
                            kalman(pxyt,xxyt,ptargetx[trackn[i]],asstargetx[assn],ptargety[trackn[i]],asstargety[assn],T,R);
                            pxy(arma::span(0,3),arma::span(i*4,i*4+3))=pxyt;
                            xxy(arma::span(0,3),arma::span(i*4,i*4+3))=xxyt;

                            pwht=pwh(arma::span(0,3),arma::span(measuren2[i]*4,measuren2[i]*4+3));
                            xwht=xwh(arma::span(0,3),arma::span(measuren2[i]*4,measuren2[i]*4+3));
                            kalman(pwht,xwht,ptargetw[trackn[i]],asstargetw[assn],ptargeth[trackn[i]],asstargeth[assn],T,R);
                            pxy(arma::span(0,3),arma::span(measuren2[i]*4,measuren2[i]*4+3))=inp*arma::eye(4,4);
                            pwh(arma::span(0,3),arma::span(measuren2[i]*4,measuren2[i]*4+3))=inp*arma::eye(4,4);
                            pwh(arma::span(0,3),arma::span(i*4,i*4+3))=pwht;
                            xwh(arma::span(0,3),arma::span(i*4,i*4+3))=xwht;

                            ctargetx[measuren2[i]]=xxyt(0,0);
                            ctargety[measuren2[i]]=xxyt(2,2);
                            ctargetw[measuren2[i]]=xwht(0,0);
                            ctargeth[measuren2[i]]=xwht(2,2);

//                            cout<<xxyt(0,0)<< endl;
//                            cout<<xxyt(2,2)<< endl;
//                            cout<<xwht(0,0)<< endl;
//                            cout<<xwht(2,2)<< endl;
                            asstargetx[assn]=ctargetx[measuren2[i]];
                            asstargety[assn]=ctargety[measuren2[i]];
                            asstargetw[assn]=ctargetw[measuren2[i]];
                            asstargeth[assn]=ctargeth[measuren2[i]];

                            ptargetx[i]=ctargetx[measuren2[i]];
                            ptargety[i]=ctargety[measuren2[i]];
                            ptargetw[i]=ctargetw[measuren2[i]];
                            ptargeth[i]=ctargeth[measuren2[i]];


                            assn++;
                        }

//                            ptargetx[i]=asstargetx[assn];
//                            ptargety[i]=asstargety[assn];
//                            ptargetw[i]=asstargetw[assn];
//                            ptargeth[i]=asstargeth[assn];
//                            trackid[i]=asstrackid[assn];
                        else{
                            unasstrackid[unassn]=trackid[i];
//                            trackid[i]=0;
                            cout << "Unassociated Track ID: " << unasstrackid[unassn] << endl;
                            unassn++;
                        }
//                        ptargetx[i]=ctargetx[i];
//                        ptargety[i]=ctargety[i];
//                        ptargetw[i]=ctargetw[i];
//                        ptargeth[i]=ctargeth[i];
                }
                for(int i=0; i<detectcount; i++){
                    trackid[i]=asstrackid[i];
                }


//                trackid.clear();
//                trackid.insert(trackid.end(),asstrackid.begin(),asstrackid.end());
//                trackid.insert(trackid.end(),unasstrackid.begin(),unasstrackid.end());
//                ptargetx.insert(ptargetx.end(),asstargetx.begin(),asstargetx.end());
//                ptargetx.insert(ptargetx.end(),unasstargetx.begin(),unasstargetx.end());
//                ptargety.insert(ptargety.end(),asstargety.begin(),asstargety.end());
//                ptargety.insert(ptargety.end(),unasstargety.begin(),unasstargety.end());
//                ptargetw.insert(ptargetw.end(),asstargetw.begin(),asstargetw.end());
//                ptargetw.insert(ptargetw.end(),unasstargetw.begin(),unasstargetw.end());
//                ptargeth.insert(ptargeth.end(),asstargeth.begin(),asstargeth.end());
//                ptargeth.insert(ptargeth.end(),unasstargeth.begin(),unasstargeth.end());
            }
            if(detectcount>pastdetectcount){
                int assn=0;
                int unassn=0;
                vector<int> asstemp(pastdetectcount);
                for(int i=0; i<pastdetectcount; i++){
//                        if(i < pastdetectcount){
                            asstrackid[assn]=trackid[trackn[i]];
                            asstargetx[assn]=ctargetx[measuren[i]];
                            asstargety[assn]=ctargety[measuren[i]];
                            asstargetw[assn]=ctargetw[measuren[i]];
                            asstargeth[assn]=ctargeth[measuren[i]];
                            asstemp[i]=measuren[i];
                            cout << "Target at (" << asstargetx[assn] << "," << asstargety[assn] << "); ";
                            cout << "Associate with Track ID: " << asstrackid[assn] << endl;



                            pxyt=pxy(arma::span(0,3),arma::span(measuren[i]*4,measuren[i]*4+3));
                            xxyt=xxy(arma::span(0,3),arma::span(measuren[i]*4,measuren[i]*4+3));
                            kalman(pxyt,xxyt,ptargetx[trackn[i]],asstargetx[assn],ptargety[trackn[i]],asstargety[assn],T,R);
                            pxy(arma::span(0,3),arma::span(i*4,i*4+3))=pxyt;
                            xxy(arma::span(0,3),arma::span(i*4,i*4+3))=xxyt;

                            pwht=pwh(arma::span(0,3),arma::span(measuren[i]*4,measuren[i]*4+3));
                            xwht=xwh(arma::span(0,3),arma::span(measuren[i]*4,measuren[i]*4+3));
                            kalman(pwht,xwht,ptargetw[trackn[i]],asstargetw[assn],ptargeth[trackn[i]],asstargeth[assn],T,R);
                            pxy(arma::span(0,3),arma::span(measuren[i]*4,measuren[i]*4+3))=inp*arma::eye(4,4);
                            pwh(arma::span(0,3),arma::span(measuren[i]*4,measuren[i]*4+3))=inp*arma::eye(4,4);
                            pwh(arma::span(0,3),arma::span(i*4,i*4+3))=pwht;
                            xwh(arma::span(0,3),arma::span(i*4,i*4+3))=xwht;

                            ctargetx[measuren[i]]=xxyt(0,0);
                            ctargety[measuren[i]]=xxyt(2,2);
                            ctargetw[measuren[i]]=xwht(0,0);
                            ctargeth[measuren[i]]=xwht(2,2);
//
//                            cout<<xxyt(0,0)<< endl;
//                            cout<<xxyt(2,2)<< endl;
//                            cout<<xwht(0,0)<< endl;
//                            cout<<xwht(2,2)<< endl;
                            asstargetx[assn]=ctargetx[measuren[i]];
                            asstargety[assn]=ctargety[measuren[i]];
                            asstargetw[assn]=ctargetw[measuren[i]];
                            asstargeth[assn]=ctargeth[measuren[i]];


                            ptargetx[i]=asstargetx[assn];
                            ptargety[i]=asstargety[assn];
                            ptargetw[i]=asstargetw[assn];
                            ptargeth[i]=asstargeth[assn];
                            trackid[i]=asstrackid[assn];

                            assn++;
//                        }

//                            unasstargetx[unassn]=ctargetx[i];
//                            unasstargety[unassn]=ctargety[i];
//                            unasstargetw[unassn]=ctargetw[i];
//                            unasstargeth[unassn]=ctargeth[i];
//
//                            trackidcount++;
//                            newtrackid[unassn]=trackidcount;
//                            cout << "Unassociated Targets: (" << unasstargetx[unassn] << "," << unasstargety[unassn] << ");";
//                            cout << "Initialise new tracks: " << trackidcount << ";";
//                            unassn++;
//                            cout << asstemp[i] << endl;

                }
                int pft=maxdetect;
                for(int k=0; k<(detectcount-pastdetectcount); k++){
                    int ft=findmissing(asstemp,k-1);
                    if(pft==ft){
                        ft++;
                    }
//                    cout <<"ft   "<< ft << endl;

                    unasstargetx[k]=ctargetx[ft];
                    unasstargety[k]=ctargety[ft];
                    unasstargetw[k]=ctargetw[ft];
                    unasstargeth[k]=ctargeth[ft];

                    trackidcount++;
                    newtrackid[k]=trackidcount;
                    cout << "Unassociated Targets: (" << unasstargetx[k] << "," << unasstargety[k] << ");";
                    cout << "Initialise new tracks: " << trackidcount << endl;
//                    unassn=detectcount-pastdetectcount;
//                    unassn++;
                    ptargetx[pastdetectcount+k]=unasstargetx[k];
                    ptargety[pastdetectcount+k]=unasstargety[k];
                    ptargetw[pastdetectcount+k]=unasstargetw[k];
                    ptargeth[pastdetectcount+k]=unasstargeth[k];
                    trackid[pastdetectcount+k]=newtrackid[k];
                    pft=ft;

                    xxy(0,4*(pastdetectcount+k))=ptargetx[pastdetectcount+k];
                    xxy(2,4*(pastdetectcount+k)+2)=ptargety[pastdetectcount+k];
                    xwh(0,4*(pastdetectcount+k))=ptargetw[pastdetectcount+k];
                    xwh(2,4*(pastdetectcount+k)+2)=ptargeth[pastdetectcount+k];


                    arma::uvec mintentrack=find(tentrackid==0);
//cout<<arma::min(mintentrack)<<endl;
                    tentrackid(0,arma::min(mintentrack))=newtrackid[k];
//                    cout << tentrackid<<endl;
                }
//                cout << " " << ptargetx[0]<< " " << ptargetx[1]<<" " << ptargetx[2]<<endl;
//                trackid.clear();
//                trackid.insert(trackid.end(),asstrackid.begin(),asstrackid.end());
//                trackid.insert(trackid.end(),newtrackid.begin(),newtrackid.end());
//                ptargetx.insert(ptargetx.end(),asstargetx.begin(),asstargetx.end());
//                ptargetx.insert(ptargetx.end(),unasstargetx.begin(),unasstargetx.end());
//                ptargety.insert(ptargety.end(),asstargety.begin(),asstargety.end());
//                ptargety.insert(ptargety.end(),unasstargety.begin(),unasstargety.end());
//                ptargetw.insert(ptargetw.end(),asstargetw.begin(),asstargetw.end());
//                ptargetw.insert(ptargetw.end(),unasstargetw.begin(),unasstargetw.end());
//                ptargeth.insert(ptargeth.end(),asstargeth.begin(),asstargeth.end());
//                ptargeth.insert(ptargeth.end(),unasstargeth.begin(),unasstargeth.end());
//                cout << " " << ptargetx[0]<< " " << ptargetx[1]<<" " << ptargetx[2]<<endl;
            }
            if(detectcount == pastdetectcount){
                int assn=0;
                for(int i=0; i<detectcount; i++){
//                        if(i == measuren[i]){
//cout << measuren[i]<< endl;
                            asstrackid[assn]=trackid[trackn[i]];
                            asstargetx[assn]=ctargetx[measuren[i]];
                            asstargety[assn]=ctargety[measuren[i]];
                            asstargetw[assn]=ctargetw[measuren[i]];
                            asstargeth[assn]=ctargeth[measuren[i]];
                            cout << "Target at (" << asstargetx[assn] << "," << asstargety[assn] << "); ";
                            cout << "Associate with Track ID: " << asstrackid[assn] << endl;



                            pxyt=pxy(arma::span(0,3),arma::span(measuren[i]*4,measuren[i]*4+3));
                            xxyt=xxy(arma::span(0,3),arma::span(measuren[i]*4,measuren[i]*4+3));
                            kalman(pxyt,xxyt,ptargetx[trackn[i]],asstargetx[assn],ptargety[trackn[i]],asstargety[assn],T,R);
                            pxy(arma::span(0,3),arma::span(i*4,i*4+3))=pxyt;
                            xxy(arma::span(0,3),arma::span(i*4,i*4+3))=xxyt;

                            pwht=pwh(arma::span(0,3),arma::span(measuren[i]*4,measuren[i]*4+3));
                            xwht=xwh(arma::span(0,3),arma::span(measuren[i]*4,measuren[i]*4+3));
                            kalman(pwht,xwht,ptargetw[trackn[i]],asstargetw[assn],ptargeth[trackn[i]],asstargeth[assn],T,R);
                            pxy(arma::span(0,3),arma::span(measuren[i]*4,measuren[i]*4+3))=inp*arma::eye(4,4);
                            pwh(arma::span(0,3),arma::span(measuren[i]*4,measuren[i]*4+3))=inp*arma::eye(4,4);
                            pwh(arma::span(0,3),arma::span(i*4,i*4+3))=pwht;
                            xwh(arma::span(0,3),arma::span(i*4,i*4+3))=xwht;

                            ctargetx[measuren[i]]=xxyt(0,0);
                            ctargety[measuren[i]]=xxyt(2,2);
                            ctargetw[measuren[i]]=xwht(0,0);
                            ctargeth[measuren[i]]=xwht(2,2);
//
//
//                            cout<<xxyt(0,0)<< endl;
//                            cout<<xxyt(2,2)<< endl;
//                            cout<<xwht(0,0)<< endl;
//                            cout<<xwht(2,2)<< endl;
                            asstargetx[assn]=ctargetx[measuren[i]];
                            asstargety[assn]=ctargety[measuren[i]];
                            asstargetw[assn]=ctargetw[measuren[i]];
                            asstargeth[assn]=ctargeth[measuren[i]];

//                            trackid[i]=asstrackid[assn];
                            ptargetx[i]=asstargetx[assn];
                            ptargety[i]=asstargety[assn];
                            ptargetw[i]=asstargetw[assn];
                            ptargeth[i]=asstargeth[assn];
//                            cout << trackid[i] << endl;
                            assn++;
//                        }

                }
                for(int i=0; i<detectcount; i++){
                    trackid[i]=asstrackid[i];
                }
//                trackid.clear();
//                cout << " " << ptargetx[0]<< " " << ptargetx[1]<<" " << ptargetx[2]<<endl;
//                trackid.insert(trackid.end(),asstrackid.begin(),asstrackid.end());
//                trackid.insert(trackid.end(),unasstrackid.begin(),unasstrackid.end());
//                ptargetx.insert(ptargetx.end(),asstargetx.begin(),asstargetx.end());
//                ptargetx.insert(ptargetx.end(),unasstargetx.begin(),unasstargetx.end());
//                ptargety.insert(ptargety.end(),asstargety.begin(),asstargety.end());
//                ptargety.insert(ptargety.end(),unasstargety.begin(),unasstargety.end());
//                ptargetw.insert(ptargetw.end(),asstargetw.begin(),asstargetw.end());
//                ptargetw.insert(ptargetw.end(),unasstargetw.begin(),unasstargetw.end());
//                ptargeth.insert(ptargeth.end(),asstargeth.begin(),asstargeth.end());
//                ptargeth.insert(ptargeth.end(),unasstargeth.begin(),unasstargeth.end());
//                cout << " " << ptargetx[0]<< " " << ptargetx[1]<<" " << ptargetx[2]<<endl;
            }
            }


            for(int i=0; i<maxdetect; i++){
                mattrackid(0,i)=trackid[i];
                matctargetx(0,i)=ctargetx[i];
                matctargety(0,i)=ctargety[i];
                matctargetw(0,i)=ctargetw[i];
                matctargeth(0,i)=ctargeth[i];
            }







            int comwindow = 5;
            int comreq = 3;
//            float statreq = comreq/comwindow;
//            cout << statreq << endl;
            for(int k=0; k<detectcount; k++){
//                trackhist[(framecount-1)*maxdetect+k]=trackid[k];
                trackhist(framecount-1,k)=trackid[k];

                targetxhist(framecount-1,k)=ptargetx[k];
                targetyhist(framecount-1,k)=ptargety[k];
                targetwhist(framecount-1,k)=ptargetw[k];
                targethhist(framecount-1,k)=ptargeth[k];

            }

            int frameb=framecount-1-comwindow;
            if(frameb<0){frameb=0;}
            arma::mat tracktest=trackhist(arma::span(frameb,framecount-1),arma::span(0,maxdetect-1));
            arma::uvec tentrackn=arma::find(tentrackid>0);

            for(int i=0; i<tentrackn.n_rows; i++){
                int tencount=0;
                for(int k=0; k<framecount-1-frameb; k++){
                    for(int kk=0; kk<maxdetect; kk++){
                        if(tracktest(k,kk)==tentrackid(0,i)){
                            tencount++;
                            tentrackcount(0,i)=tencount;
                        }
                    }
                }
            }

            arma::uvec comtrackat=arma::find(tentrackcount>=comreq);
//            cout << comtrackat<<endl;
            for(int i=0; i<comtrackat.n_rows; i++){
                comtrackid(0,arma::min(arma::find(comtrackid==0)))=tentrackid(0,comtrackat(i,0));
                tentrackid(0,comtrackat(i,0))=0;
            }
//            cout <<tentrackid<<endl;
//            cout<<comtrackid<<endl;
            comhist(framecount-1,arma::span(0,maxdetect-1))=comtrackid;
//            for(int k=0; k<comtrackat.n_rows; k++){
//                comhist(framecount-1,k)=comtrackid(0,k);
//            }

            if(framecount-1-frameb==5){
                arma::uvec uptotentrackid=arma::find(tentrackid>0);

                arma::uvec nocomtrackat=arma::find(tentrackcount(0,arma::span(0,arma::max(uptotentrackid)))<comreq);

//                arma::min(arma::find(nocomtrackhist==0));

                for(int i=0; i<nocomtrackat.n_rows; i++){
//                    tentrackid(0,nocomtrackat(i,0))=0;
                    arma::uvec check=arma::find(nocomtrackhist.row(0)==nocomtrackat(i,0));
                    if(!check.is_empty()){
                        nocomtrackhist(1,check(0,0))++;
                        break;
                    }
                    else{
                        nocomtrackhist(0,arma::min(arma::find(nocomtrackhist.row(0)==0)))=tentrackid(0,nocomtrackat(i,0));
                        nocomtrackhist(1,arma::min(arma::find(nocomtrackhist.row(0)==0)))=nocomtrackhist(1,arma::min(arma::find(nocomtrackhist.row(0)==0)))+1;
                    }

                }
//                cout << nocomtrackhist<<endl;
                for(int k=0; k<maxdetect; k++){
                        if(nocomtrackhist(0,k)>0.1){
                            if(nocomtrackhist(1,k)>=3){
                            tentrackid(0,nocomtrackhist(0,k))=0;
                            nocomtrackhist(0,k)=0;
                            nocomtrackhist(1,k)=0;
                            }
                        }
                }
            }

//            cout << tentrackid<<endl;
            arma::mat comtest=comhist(arma::span(frameb,framecount-1),arma::span(0,maxdetect-1));
            arma::uvec comtrackn=arma::find(comtrackid>0);

//            cout << comtrackn<<endl;
            while(!comtrackn.is_empty()){
//                cout << comtest << endl;
                for(int i=0; i<comtrackn.n_rows; i++){
                    int comcount=0;
                    for(int k=0; k<framecount-1-frameb; k++){
                        for(int kk=0; kk<maxdetect; kk++){
                            if(comtest(k,kk)==comtrackid(0,i)){
                                comcount++;
                                comtrackcount(0,i)=comcount;

                            }
                        }
                    }
                }
//                cout << comtrackcount <<endl;
                break;
            }


//            if(comcount>0){
//                arma::uvec termtrackat=arma::find(comtrackcount<comreq);
//                cout << termtrackat.n_rows<< endl;
//                for(int k=0; k<termtrackat.n_rows; k++){
//                    termtrackhist(framecount-1,k)=comtrackid(0,termtrackat(k,0));
//                }
//                cout << "HERE"<< endl;
//                for(int i=0; i<termtrackat.n_rows; i++){
//                    comtrackid(0,termtrackat(i,0))=0;
//
//                }
//            }




//            for(int k=0; k<comtrackat.n_rows; k++){
//                cout << mattrackid << endl;
//                cout << comtrackid << endl;
                char idname[maxdetect];
                arma::uvec findid=arma::find(comtrackid>0);
                for(int i=0; i<findid.n_rows; i++){
                    sprintf(idname,"ID: %i",(int)comtrackid(0,i));
                    arma::uvec findtarget=arma::find(mattrackid==comtrackid(0,i));

                    if(findtarget.is_empty()){
                        arma::uvec findpast=arma::find(trackhist==comtrackid(0,i));


                        Point idplace(targetxhist(arma::max(findpast)),targetyhist(arma::max(findpast))-5);
                        putText(framemog2,idname,idplace,FONT_HERSHEY_SIMPLEX,0.5,Scalar(255,0,0),1);

                        Rect trackbounds;
                        trackbounds.x=targetxhist(arma::max(findpast));
                        trackbounds.y=targetyhist(arma::max(findpast));
                        trackbounds.width=targetwhist(arma::max(findpast));
                        trackbounds.height=targethhist(arma::max(findpast));
                        rectangle(framemog2, trackbounds, cv::Scalar(255,0,0),1);

                        comtrackid(0,i)=0;
                    }
                    else{

//                    cout <<mattrackid(findtarget(0))<<endl;
                    Point idplace(matctargetx(0,findtarget(0)),matctargety(0,findtarget(0))-5);
                    putText(framemog2,idname,idplace,FONT_HERSHEY_SIMPLEX,0.5,Scalar(0,0,255),1);

                    Rect trackbounds;
                    trackbounds.x=matctargetx(0,findtarget(0));
                    trackbounds.y=matctargety(0,findtarget(0));
                    trackbounds.width=matctargetw(0,findtarget(0));
                    trackbounds.height=matctargeth(0,findtarget(0));
                    rectangle(framemog2, trackbounds, cv::Scalar(0,0,255),1);
//                    cout << trackbounds.width << "   "<< trackbounds.height<<endl;
                    }
//                  cout<<comtrackid<<endl;

                }
            imshow("OutputMOG2",framemog2);
            pastdetectcount=detectcount;
        }



        char filename[100];
        sprintf(filename,"../result/indoorpeople/frame%i.jpg",framecount);
        imwrite(filename,framemog2);


        if(waitKey(5) == 'p'){
            pauseplay = !pauseplay;
            if(pauseplay == false){
                cout << "Video Paused" << endl;
            }
            while(!pauseplay){
                if(waitKey(5) == 'p'){
                    pauseplay = !pauseplay;
                }
                if(waitKey(30) == 27){
                    abort();
                }
            }
            if(pauseplay == true){
                cout << "Video Continues" << endl;
            }
        }

        if(waitKey(30) == 27) //wait for 'esc' key press for 30 ms. If 'esc' key is pressed, break loop
       {
                cout << "esc key is pressed by user" << endl;
                abort();
       }


//       while(true){
//            if(waitKey(5) == 'c'){
//                break;
//            }
//       }

        tentrackcount.fill(0);
        comtrackcount.fill(0);
        matctargetx.fill(0);
        matctargety.fill(0);
        matctargeth.fill(0);
        matctargetw.fill(0);


        framecount++;
        waitKey(100);
    }
    return 0;
}
