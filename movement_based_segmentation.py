import cv2
import os
import numpy as np
from yoloseg import utils
import argparse

parser=argparse.ArgumentParser(description="A program to take an input fall video from he stream and transform it in an output that the users can rec")
parser=argparse.ArgumentParser(description="A program to take an input fall video from he stream and transform it in an output that the users can recognize")
parser.add_argument("--video_path", help="Path of the video you want to process use / to separate folders", default=None, type=str)
parser.add_argument("--output_path", help="Output path o the video result", type=str)
parser.add_argument("--fps", help="Frames per second of the resuling video", type=int)
parser.add_argument("--see_result", help="Visualize the results inmediately after prorcessing", default=False, action="store_true")
parser.add_argument("--Kernel_size", help="The kernel size for the morological closure operation", type=int, default=100)
args=parser.parse_args()

video_path=args.video_path
output_path=args.output_path

cap = cv2.VideoCapture(video_path)

frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height= cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
kernel=np.ones((args.Kernel_size,args.Kernel_size), np.uint8)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, args.fps, (int(frame_width), int(frame_height)))

done, CurrentFrame = cap.read()
done, NextFrame = cap.read()

if args.see_result:
    cv2.namedWindow("Detected objects", cv2.WINDOW_NORMAL)

while cap.isOpened():
    if done==True:
        diff = cv2.absdiff(CurrentFrame, NextFrame)
        
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        blured_img = cv2.GaussianBlur(gray, (5, 5), 0)
        
        threshold, binary_img = cv2.threshold(blured_img, 35, 255, cv2.THRESH_BINARY)
        
        binary_img_closed=cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
        
        dilated = cv2.dilate( binary_img_closed, None, iterations=30)

        dilated = cv2.GaussianBlur(dilated, (0,0), sigmaX=10, sigmaY=10)
        dilated=cv2.erode(dilated, kernel=kernel)
        
        contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_NONE)
        
        
        #create an color image from the person mask
        image_color=np.zeros((dilated.shape[0], dilated.shape[1],3),dtype=np.uint8)
        image_color[:,:]=utils.colors[0]
        image_color=cv2.bitwise_and(image_color, image_color, mask=dilated)
        #obscuring orignal frame
        canny=cv2.Canny(CurrentFrame, 75,75)
        canny=cv2.GaussianBlur(canny, (0,0), sigmaX=10)
        canny=cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
        final_image=cv2.bitwise_or(canny, image_color)
        final_image=cv2.bilateralFilter(final_image, d=2, sigmaColor=0.5, sigmaSpace=5)
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            
            if cv2.contourArea(contour) < 1000:
                continue
            
            cv2.rectangle(final_image, (x, y), (x+w, y+h), (0, 0, 255), 2)
        if args.see_result:
            cv2.imshow("Detected objects", final_image)
        #cv2.imshow("binary_img", cv2.flip(binary_img_closed, 1))
        #dilated_blurred=cv2.GaussianBlur(dilated, (0,0), sigmaX=15, sigmaY=15) 
        #cv2.imshow("dilated_img", dilated_blurred)
        
        out.write(final_image)
        
        CurrentFrame = NextFrame
        
        done, NextFrame = cap.read()
        
        if cv2.waitKey(30) == ord("g"):
            break
    else: break
cv2.destroyAllWindows()
cap.release()
out.release()