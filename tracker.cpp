#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <iostream>
#include <math.h>
#include <stddef.h>


using namespace cv;
using namespace std;


// Specifying minimum and maximum size parameters
#define MIN_FACE_SIZE 24
#define MAX_FACE_SIZE 100

#define SKIPFRAME 30
#define RATIO 1


int main(int argc, char **argv)
{
    std::vector<cv::Point2f> last_object;
    for (int i = 0; i < 68; ++i) {
        last_object.push_back(cv::Point2f(0.0, 0.0));
    }

    try
    {

        //-------------------------------------face detector-----------------------------
        // Load the Cascade Classifier Xml file
        CascadeClassifier faceCascade("cascade/haarcascade_lefteye_2splits.xml");

        // Create a VideoCapture object
        VideoCapture cap(0);

        // Check if camera opened successfully
        if (!cap.open(0)) return 0;

        Mat frame, frameBig, frameGray;
        //-----------------------------------face detector end------------------------------

        //----------------------------------------tracker------------------------------------
        // List of tracker types in OpenCV 3.2
        // NOTE : GOTURN implementation is buggy and does not work.
        const char *types[] = { "BOOSTING", "MIL", "KCF", "TLD","MEDIANFLOW", "GOTURN" };
        std::vector <string> trackerTypes(types, std::end(types));

        // Create a tracker
        string trackerType = trackerTypes[2];
        Ptr<Tracker> tracker_0 = Tracker::create(trackerType);
        Ptr<Tracker> tracker_1 = Tracker::create(trackerType);

        // Define initial boundibg box
        Rect2d bbox(10, 10,10, 10);
        Rect2d bbox_0(10, 10,10, 10);
        Rect2d bbox_1(10, 10,10, 10);
        Rect2d facebox(10, 10,10, 10);

        int contframe = 0;
        //----------------------------tracker end------------------------------------------
        unsigned long im_mum=0;

        Mat grayscale_image, img_adaptive,img_EqualizeHist;

        // Creating vector to store the detected faces' parameters
        vector<Rect> faces;
        bool ok_0=0;
        bool ok_1=0;
        while (1)
        {
            // Reading each frame
            bool frameRead = cap.read(frameBig);

            // If frame not opened successfully
            if (!frameRead)
                break;


            cv::resize(frameBig, frame, cv::Size(), 1.0 / RATIO, 1.0 / RATIO);

            dlib::cv_image<dlib::bgr_pixel> cimg(frame);
            dlib::cv_image<dlib::bgr_pixel> cframeBig(frameBig);

            // Detect faces
            faceCascade.detectMultiScale(frame, faces, 1.1, 5, 0, Size(MIN_FACE_SIZE, MIN_FACE_SIZE), Size(MAX_FACE_SIZE, MAX_FACE_SIZE));

            contframe++;
            if(contframe > SKIPFRAME)
                contframe =0;

            //printf("contframe is :%d\n",contframe);

            if(contframe == 0)
            {
                // Loop over each detected face
                for (int i = 0; i < faces.size(); i++)
                {
                    // Dimension parameters for bounding rectangle for face
                    Rect faceRect = faces[i];

                    if(i==0)
                    {
                        bbox_0 = faceRect;
                        tracker_0 = Tracker::create(trackerType);
                        tracker_0->init(frame, bbox_0);
                        ok_0 = tracker_0->update(frame, bbox_0);

                    }

//                    if(i==1)
//                    {
//                        bbox_1 = faceRect;
//                        tracker_1 = Tracker::create(trackerType);
//                        tracker_1->init(frame, bbox_1);
//                        ok_1=tracker_1->update(frame, bbox_1);

//                    }

                }
            }

            if(ok_0 )
            {
                  tracker_0->update(frame, bbox_0);
                  rectangle(frame,bbox_0, Scalar(128, 255, 0), 2);

            }
            if(ok_1 )
            {
                    tracker_1->update(frame, bbox_1);
                    rectangle(frame,bbox_1, Scalar(128, 255, 0), 2);
            }
            imshow("tracking", frame);
            // Exit if ESC pressed.
            int k = waitKey(1);
            if (k == 27)
            {
                break;
            }
        }
    }
    catch (exception& e)
    {
        cout << e.what() << endl;
    }
    getchar();

    return 0;
}
