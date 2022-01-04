import cv2
import copy
from detectors import Detectors
from tracker import Tracker


def main():
    # Create opencv video capture object
    # cap = cv2.VideoCapture('data/TrackingBugs.mp4')
    cap = cv2.VideoCapture('video_3_bin.mp4')

    # Create Object Detector
    detector = Detectors()

    # Create Object Tracker
    tracker = Tracker(160, 20, 50, 100)  # 追踪过程中各种阈值参数

    # Variables initialization
    skip_frame_count = 0
    track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                    (0, 255, 255), (255, 0, 255), (255, 127, 255),
                    (127, 0, 255), (127, 0, 127)]
    pause = False

    frame_count = 1
    # Infinite loop to process video frames
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()  # ret是一个bool值，如果是true，表示帧读取是正确的
        # frame表示截取到的一帧的图片的数据，是三维数组
        # Make copy of original frame
        orig_frame = copy.copy(frame)




        # Convert binary image to greyscale
        if ret:
           frame[frame != 0] = 255

        # Skip initial frames that display logo
        if skip_frame_count < 1:
            skip_frame_count += 1
            continue
        print(ret)
        # Detect and return centroids of the objects in the frame
        if ret:
            print("Processing frame " + format(frame_count))
            frame_count += 1
            # if frame_count == 5:
            #     a = orig_frame
               # cv2.imwrite('5.jpg', a)
            centers = detector.Detect(frame, frame_count)


            # If centroids are detected then track them
            if len(centers) > 0:

                # Track object using Kalman Filter
                tracker.Update(centers)

                # For identified object tracks draw tracking line
                # Use various colors to indicate different track_id
                for i in range(len(tracker.tracks)):
                    if len(tracker.tracks[i].trace) > 1:
                        for j in range(len(tracker.tracks[i].trace)-1):
                            # Draw trace line
                            x1 = tracker.tracks[i].trace[j][0][0]
                            y1 = tracker.tracks[i].trace[j][1][0]
                            x2 = tracker.tracks[i].trace[j+1][0][0]
                            y2 = tracker.tracks[i].trace[j+1][1][0]
                            clr = tracker.tracks[i].track_id % 9
                            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                     track_colors[clr], 2)

                # Display the resulting tracking frame
                cv2.imshow('Tracking', frame)

            # Display the original frame
            cv2.imshow('Original', orig_frame)

            # Slower the FPS
            cv2.waitKey(50)

            # # # Check for key strokes
            # k = cv2.waitKey(50) & 0xff
            # if k == 27:  # 'esc' key has been pressed, exit program.
            #     break
            # if k == 112:  # 'p' has been pressed. this will pause/resume the code.
            #     pause = not pause
            #     if pause is True:
            #         print "Code is paused. Press 'p' to resume.."
            #         while pause is True:
            #             # stay in this loop until
            #             key = cv2.waitKey(30) & 0xff
            #             if key == 112:
            #                 pause = False
            #                 print "Resume code..!!"
            #                 break
        else:
            print("All frames were processed")
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # execute main
    main()
