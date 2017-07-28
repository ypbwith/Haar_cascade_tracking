import cv2
import sys
import numpy as np
import numpy
import mahotas

if __name__ == '__main__' :

    # Set up tracker.
    # Instead of MIL, you can also use

    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
    tracker_type = tracker_types[2]
    tracker = cv2.Tracker_create(tracker_type)

    # Read video
    # video = cv2.VideoCapture("videos/chaplin.mp4")
    cap = cv2.VideoCapture(0)

    ok, frame = cap.read()  # 读取一桢图像，前一个返回值是是否成功，后一个返回值是图像本身
    # Converting to grayscale
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # # Exit if video not opened.
    # if not video.isOpened():
    #     print "Could not open video"
    #     sys.exit()
    #
    # # Read first frame.
    # ok, frame = video.read()
    # if not ok:
    #     print 'Cannot read video file'
    #     sys.exit()

    # # Define an initial bounding box
    # bbox = (287, 23, 86, 320)

    # Uncomment the line below to select a different bounding box
    # bbox = cv2.selectROI(frame, False)


    # Load the Cascade Classifier Xml file
    MIN_FACE_SIZE = 100
    MAX_FACE_SIZE = 300
    face_cascade = cv2.CascadeClassifier("cascade/mallick_haarcascade_frontalface_default.xml")
    # Specifying minimum and maximum size parameters
    # Detect faces
    faceRects = face_cascade.detectMultiScale(frameGray, 2,20, 0, (MIN_FACE_SIZE, MIN_FACE_SIZE),
                                              (MAX_FACE_SIZE, MAX_FACE_SIZE))

    # if faceRects is not None:
    #     # Loop over each detected face
    #     for faceRect in faceRects:  # 对每一个人脸画矩形框
    #         x, y, w, h = faceRect
    #         bbox = (x, y, w, h)
    #         cv2.rectangle(frameGray, (x, y), (x + w, y + h), (255, 0, 0))
    #         # Dimension parameters for bounding rectangle for face
    #
    #         # Calculating the dimension parameters for eyes from the dimensions parameters of the face
    #         ex, ey, ewidth, eheight = int(x + 0.125 * w), int(y + 0.25 * h), int(0.75 * w), int(0.25 * h)
    #
    #         # Drawing the bounding rectangle around the face
    #         cv2.rectangle(frame, (ex, ey), (ex + ewidth, ey + eheight), (128, 255, 0), 2)
    #         # Initialize tracker with first frame and bounding box
    #
    # ok = tracker.init(frame, bbox)
    bbox = None;
    countfream = 0;

    while True:
        countfream= countfream+1
        if countfream is 10:
            countfream = 0
        # Read a new frame
        ok, frame = cap.read()
        if not ok:
            break
        # Converting to grayscale
        frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if bbox is None or countfream == 0 :
            faceRects = face_cascade.detectMultiScale(frameGray, 1.1, 5, 0, (MIN_FACE_SIZE, MIN_FACE_SIZE),
                                                      (MAX_FACE_SIZE, MAX_FACE_SIZE))
            bbox = (0, 0, 0, 0)
            if faceRects is not None:
                # Loop over each detected face
                for faceRect in faceRects:  # 对每一个人脸画矩形框
                    x, y, w, h = faceRect
                    bbox = (x, y, w, h)
                    # print(bbox)
                    cv2.rectangle(frameGray, (x, y), (x + w, y + h), (255, 0, 0))
                    # Dimension parameters for bounding rectangle for face
                    # Calculating the dimension parameters for eyes from the dimensions parameters of the face
                    # ex, ey, ewidth, eheight = int(x + 0.125 * w), int(y + 0.25 * h), int(0.75 * w), int(0.25 * h)
                    # bbox_eye = (ex, ey,  ewidth, eheight)
                    # # Drawing the bounding rectangle around the face
                    # cv2.rectangle(frame, (ex, ey), (ex + ewidth, ey + eheight), (128, 255, 0), 2)
                    # Initialize tracker with first frame and bounding box
                    tracker = cv2.Tracker_create(tracker_type)
                    ok = tracker.init(frame, bbox)

        # Start timer
        timer = cv2.getTickCount()
        # Update tracker
        ok, bbox = tracker.update(frame)

        ex, ey, ewidth, eheight = int(bbox[0] + 0.125 * bbox[2]), int(bbox[1] + 0.25 * bbox[3]), int(0.75 * bbox[2]), int(0.25 * bbox[3])
        # bbox_eye = (ex, ey, ewidth, eheight)
        bbox_eye = bbox ;
        pixel_data = np.array(frameGray)
        sp =frameGray.shape
        dst_X = int( bbox_eye[0])
        dst_Y = int( bbox_eye[1])
        Delta_X = int(bbox_eye[0] + bbox_eye[2])
        Delta_Y = int(bbox_eye[1] + bbox_eye[3])
        if (dst_X <=0 or dst_X >= sp[1] ):
            dst_X = 0
            Delta_X =1
        if (dst_Y <= 0 or dst_Y >= sp[0] ):
            dst_Y = 0
            Delta_Y =1
        if ( Delta_Y >= sp[0]):
            dst_Y = 0
            Delta_Y = 1
        if (Delta_X >= sp[1]):
            dst_X = 0
            Delta_X = 1
        dstframe = pixel_data[ dst_Y: Delta_Y  , dst_X:Delta_X]

        img_gaussian = cv2.GaussianBlur( frameGray, (5, 5), 3)

        canny = cv2.Canny(img_gaussian, 60,90)

        # circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT, 1, 20, 30, 15, 0, 0)


        #  Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        else :
            # Tracking failure
            # ok=0
            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);

        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);


        # Display result
        cv2.imshow("Tracking", frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break

        # Display result
        cv2.imshow("dstframe", numpy.hstack([canny]))

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27: break
# Release VideoCapture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()