import cv2
import os
import numpy as np
from yoloseg import utils
import glob
import time

#from imread_from_url import imrea
# d_from_url

from yoloseg import YOLOSeg
from scipy.ndimage import gaussian_filter

# Initialize YOLOv5 Instance Segmentator
start=time.time()
model_path = "models/yolov8m-seg.onnx"
yoloseg = YOLOSeg(model_path, conf_thres=0.5, iou_thres=0.3)

# Read image
img_url = "data\images_3"
images_paths=glob.glob('data/images_3/*.jpg')

# read video frames
frames_processed=np.empty((226, 1080, 1920,3), dtype=np.uint8)
frame_count=0

for path in images_paths:

    img = cv2.imread(path)



    # Detect Objects
    start=time.time()
    boxes, scores, class_ids, masks = yoloseg(img)
    print(time.time()-start)
    # only color the segmentation if a human was detected
    if 0 in class_ids:
        masks_dict=dict(zip(class_ids, masks))
        # extract only person mask
        person_mask=masks_dict[0].astype(np.uint8)
        #create an color image from the person mask
        image_color=np.zeros((person_mask.shape[0], person_mask.shape[1],3),dtype=np.uint8)
        image_color[:,:]=utils.colors[0]
        image_color=cv2.bitwise_and(image_color, image_color, mask=person_mask)
        #obscuring orignal frame
        canny=cv2.Canny(img, 75,75)
        canny=cv2.GaussianBlur(canny, (0,0), sigmaX=10)
        canny=cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
        final_image=cv2.bitwise_or(canny, image_color)
        final_image=cv2.bilateralFilter(final_image, d=2, sigmaColor=1.5, sigmaSpace=1.5)
        frames_processed[frame_count]=final_image
        frame_count += 1
    else:
        #obscuring orignal frame
        canny=cv2.Canny(img, 75,75)
        canny=cv2.GaussianBlur(canny, (0,0), sigmaX=10)
        canny=cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
        canny=cv2.bilateralFilter(canny, d=2, sigmaColor=1.5, sigmaSpace=1.5)
        frames_processed[frame_count]=canny
        frame_count += 1

# post processing of the frames with temporal filter

filtered_frames=gaussian_filter(frames_processed, sigma=(1,0,0,0))


# write the results to a video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output_per_frame.mp4", fourcc, 15, (1920, 1080))


#visualize results inmediately
cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
for frame in filtered_frames:
    cv2.imshow("Detected Objects", frame)
    out.write(frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
out.release()
finish=time.time()

print(f'it took me {finish-start} seconds to process {226} ({226/(finish-start)} fps)')