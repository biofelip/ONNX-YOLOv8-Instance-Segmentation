import numpy as np
import cv2
from yoloseg.utils import colors
from yoloseg import YOLOSeg
from scipy.ndimage import gaussian_filter
import os
import argparse

parser=argparse.ArgumentParser(description="A program to take an input fall video from he stream and transform it in an output that the users can rec")
parser=argparse.ArgumentParser(description="A program to take an input fall video from he stream and transform it in an output that the users can recognize")
parser.add_argument("--video_path", help="Path of the video you want to process use / to separate folders", default=None, type=str)
parser.add_argument("--output_path", help="Output path o the video result", type=str)
parser.add_argument("--fps", help="Frames per second of the resuling video", type=int)
parser.add_argument("--see_result", help="Visualize the results inmediately after prorcessing", default=False, action="store_true")
parser.add_argument("--temporal_smooth", help="Should a temporal filter be applied to the video",  default=False, action="store_true")
args=parser.parse_args()

# # Initialize video
video_path=args.video_path
output_path=args.output_path

cap = cv2.VideoCapture(video_path)
fps=int(cap.get(cv2.CAP_PROP_FPS))
frames_total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
height_video=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width_video=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# Initialize YOLOv5 Instance Segmentator
model_path = "models/yolov8m-seg.onnx"
yoloseg = YOLOSeg(model_path, conf_thres=0.5, iou_thres=0.3)



frames_processed=np.empty((frames_total, height_video, width_video,3), dtype=np.uint8)
frame_count=0
# write the results to a video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, args.fps, (width_video, height_video))
# read video frames
while cap.isOpened():

    #Press key q to stop
    if cv2.waitKey(25) == ord('q'):
       break

    # Read frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    # Update object localizer
    boxes, scores, class_ids, masks = yoloseg(frame)
    # only color the segmentation if a human was detected
    if 0 in class_ids:
        masks_dict=dict(zip(class_ids, masks))
        # extract only person mask
        person_mask=masks_dict[0].astype(np.uint8)
        #create an color image from the person mask
        image_color=np.zeros((person_mask.shape[0], person_mask.shape[1],3),dtype=np.uint8)
        image_color[:,:]=colors[0]
        image_color=cv2.bitwise_and(image_color, image_color, mask=person_mask)
        #obscuring orignal frame
        canny=cv2.Canny(frame, 75,75)
        canny=cv2.GaussianBlur(canny, (0,0), sigmaX=10)
        canny=cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
        final_image=cv2.bitwise_or(canny, image_color)
        final_image=cv2.bilateralFilter(final_image, d=2, sigmaColor=1.5, sigmaSpace=1.5)
        out.write(final_image)
        frames_processed[frame_count]=final_image
        frame_count += 1
    else:
        #obscuring orignal frame
        canny=cv2.Canny(frame, 75,75)
        canny=cv2.GaussianBlur(canny, (0,0), sigmaX=10)
        canny=cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
        canny=cv2.bilateralFilter(canny, d=2, sigmaColor=1.5, sigmaSpace=1.5)
        out.write(canny)
        frames_processed[frame_count]=canny
        frame_count += 1
cap.release()
out.release()

# post processing of the frames with temporal filter
if args.temporal_smooth:
    frames_processed=gaussian_filter(frames_processed, sigma=(1,0,0,0))




if args.see_result:
    #visualize results inmediately
    cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
    for frame in frames_processed:
        cv2.imshow("Detected Objects", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
