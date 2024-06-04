import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import pandas as pd
import time
import glob
import random
from collections import Counter
import timeit
import os

start = timeit.default_timer()

def pixel_to_coord(x_good, y_good, center_x, center_y, lat_scale):
    latitude = []
    longitude = []
    for x, y in zip(x_good, y_good):
        distance = math.hypot(x - center_x, y - center_y)
        lat = distance * lat_scale
        dx = x - center_x
        dy = y - center_y
        
        # Calculate angle in radians where East (0E) corresponds to the top of the plot
        angle_rad = math.atan2(dy, dx)
        
        # Convert angle to degrees and adjust so that 0E is at 90 degrees on the regular plot
        angle_deg = math.degrees(angle_rad) + 90
        
        # Normalize the angle so that it increases counterclockwise
        if angle_deg < 0:
            angle_deg += 360
        
        # Append rounded values
        longitude.append(round(angle_deg, 3))
        latitude.append(round(-90 + lat, 3))  # Assuming this is the correct formula for latitude based on your scale
    return latitude, longitude


MY = [28,29, 30, 31]
min_area = 25
bound_per = 30
ellipse_area = 25
n = 50
size = 3

target_ls = [180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300]

for jj in range(len(MY)):
    # Initialize DataFrame for the current MY and target_Ls
    data_columns = ['x', 'y', 'latitude', 'longitude']
    df = pd.DataFrame(columns=data_columns)
    print(MY[jj])
    image_list = sorted(glob.glob('/home/pruthvi/Desktop/MARCI_VIDEOS/South_Pole/MY'+str(MY[jj])+'_South_Projected/*.jpg'))
    date = pd.read_excel('/home/pruthvi/Desktop/MARCI_VIDEOS/South_Pole/MY'+str(MY[jj])+'date.xlsx')

    dateLs = date['LS']/10

    low_H = [31,10, 24, 31]
    low_S = [0,0, 0, 0]
    low_V = [129,173, 139, 148]

    high_H = [180,180, 180,180]
    high_S = [106,145, 114, 124]
    high_V = [255,255, 255, 255]

    for ii in range(len(image_list)):
        if ii == 0:
            frames = cv2.imread(image_list[ii]) 
            heightf, widthf = frames.shape[:2]
            center_x = widthf // 2
            center_y = heightf // 2

            ##Number of backgound Points
            x_back = np.linspace(20, widthf-30, n)
            y_back = np.linspace(20, heightf, n)

            #Converting the BGR frame into an HSV
            frame_HSV = cv2.cvtColor(frames, cv2.COLOR_BGR2HSV)
            #Colour Detector
            frame_threshold = cv2.inRange(frame_HSV, (low_H[jj], low_S[jj], low_V[jj]), (high_H[jj], high_S[jj], high_V[jj]))
            #Finding the contours
            contours,hierarchy, = cv2.findContours(frame_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            #cv2.drawContours(frames, contours, -1, (0,0,255), 5)
            
            areas = [cv2.contourArea(c) for c in contours]
            sorted_areas = np.sort(areas)
            max_area = sorted_areas[-3] #the biggest contour
            #Grouping the the top 3 contours
            x_list = []
            y_list = []
            for c in contours:
                if cv2.contourArea(c) > max_area:
                    for cc in range(len(c)):
                        x_list.append(c[cc][0][0])
                        y_list.append(c[cc][0][1])
                        #cv2.circle(frames, (c[cc][0][0],c[cc][0][1]), radius=0, color=(0, 255, 0), thickness=5)
            L = [[] for i in range(len(x_list))]
            for dd in range(len(x_list)):
                L[dd] = [x_list[dd],y_list[dd]]
            cnt = np.array(L).reshape((-1,1,2)).astype(np.int32)

            ((centxo,centyo), (widtho,heighto), angleo) = cv2.fitEllipse(cnt)
            #Refitting the ellipse and only chooseing points that are on the outer edge of the 1st fit
            x_good = []
            y_good = []
            dis_good = []
            #Filteringg the contour points 
            for dd in range(len(x_list)):
                dis = math.hypot(x_list[dd] - widthf/2, y_list[dd] - heightf/2)
                cv2.circle(frames, (x_list[dd],y_list[dd]), radius=0, color=(0, 0, 255), thickness=2)
                diff_major = dis - heighto/2
                if diff_major >= 0:
                    x_good.append(x_list[dd])
                    y_good.append(y_list[dd])
                    cv2.circle(frames, (x_list[dd],y_list[dd]), radius=0, color=(0, 255, 0), thickness=2)
                    dis_good.append(math.hypot(x_list[dd] - widthf/2, y_list[dd] - heightf/2))
            #Putting the good points to into CV2 format
            L = [[] for i in range(len(x_good))]
            for dd in range(len(x_good)):
                #cv2.circle(frame, (x_good[dd],y_good[dd]), radius=0, color=(0, 0, 0), thickness=5)
                L[dd] = [x_good[dd],y_good[dd]]
            ctr = np.array(L).reshape((-1,1,2)).astype(np.int32)
            cv2.drawContours(frames,[ctr],0,(0,255,0),-1)
            #Refitting the ellipse
            ((centx,centy), (width,height), angle) = cv2.fitEllipse(ctr)
            save_file_dir = f'/home/pruthvi/Desktop/MARCI_VIDEOS/CODE/SOUTH_POLE/OUTLINES/MY_{28+jj}'
            filename = os.path.basename(image_list[ii])
            full_path = f'{save_file_dir}/{filename}'
            print(full_path)
            cv2.imwrite(full_path, frames)  # Assuming 'frames' is your image data

        else:
            print(dateLs[ii])
            frames = cv2.imread(image_list[ii])
            filename = os.path.basename(image_list[ii])
            print(image_list[ii])
            heightf, widthf = frames.shape[:2]
            #Creatingg a copy of the image for the mask image
            mask_image = frames.copy()
            
            #Creatingg a white ellipse using the previous ellipse 
            cv2.ellipse(mask_image, (int(centx),int(centy)), (int(width/2),int(height/2) ), angle, 0, 360, (255,255,255), -1)
            
            #Creating a mask usingg the white filled ellipse
            frame_gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
            retval, threshold = cv2.threshold(frame_gray, 254, 255, cv2.THRESH_BINARY)
            #Masked image
            frame_blank = cv2.bitwise_or(frames,frames,mask = threshold)

            #Finding the mean H, S and V value of the cap
            x_back = np.linspace(20, widthf - 30, n)
            y_back = np.linspace(20, heightf - 30, n)
            
            h_list = []
            s_list = []
            v_list = []
            
            for i in range(len(x_back)):
                for j in range(len(y_back)):
                    #Cropping out a small square of
                    crop = frame_blank[int(y_back[j] - int(size)):int(y_back[j] + int(size)),int(x_back[i] - int(size)):int(x_back[i] + int(size))]
                    cv2.rectangle(frame_blank,(int(x_back[i] - int(size)),int(y_back[j] - int(size))),(int(x_back[i] + int(size)),int(y_back[j] + int(size))),(0,0,255),1)
                    #Converting cropped image into a HSV image
                    crop_HSV = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                    h,s,v = cv2.split(crop)
                    if 0 not in v:
                        h_list.append(np.mean(h))
                        s_list.append(np.mean(s))
                        v_list.append(np.mean(v)) 
                        b,g,r = cv2.split(crop)
            
            mask_HSV_image = frames.copy()
            #Creatingg a white ellipse using the previous ellipse 
            cv2.ellipse(mask_HSV_image, (int(centx),int(centy)), (int(width/2) + int((width/2)* ellipse_area/100),int(height/2) + int((height/2)* ellipse_area)), angle, 0, 360, (255,255,255), -1)
            
            #Creating a mask usingg the white filled ellipse
            frame_gray = cv2.cvtColor(mask_HSV_image, cv2.COLOR_BGR2GRAY)
            retval, threshold = cv2.threshold(frame_gray, 254, 255, cv2.THRESH_BINARY)
            #Masked image
            frame_blank = cv2.bitwise_or(frames,frames,mask = threshold)  

            #Converting the frame into an hsv frame
            frame_hsv = cv2.cvtColor(frame_blank, cv2.COLOR_BGR2HSV)
            frame_threshold = cv2.inRange(frame_hsv, (23, 0, np.mean(v_list) - bound_per/100 * np.mean(v_list)), (180, np.mean(s_list) + bound_per/100 * np.mean(s_list) , 255))
            contours,hierarchy, = cv2.findContours(frame_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

            areas = [cv2.contourArea(c) for c in contours]
            if len(areas) != 0:
                sorted_areas = np.sort(areas)
                max_area = sorted_areas[-1] * min_area/100 #the biggest contour
                #Grouping the larger contours into one
                x_list = []
                y_list = []
                size_crop = 10
                for c in contours:
                    if cv2.contourArea(c) > max_area:
                        for cc in range(len(c)):
                            #Checking for black region
                            x_temp = np.int64(c[cc][0][0])
                            y_temp  = np.int64(c[cc][0][1])
                            crop2 = frames[y_temp - size_crop:y_temp + size_crop, x_temp - size_crop:x_temp + size_crop]
                            croph, cropw = crop2.shape[:2]
                            if len(crop2) != 0 and cropw != 0 and croph != 0:
                                crop_HSV2 = cv2.cvtColor(crop2, cv2.COLOR_BGR2HSV)
                                h2,s2,v2 = cv2.split(crop2)
                                if 0 not in v2: 
                                    x_list.append(x_temp)
                                    y_list.append(y_temp)
                                    #cv2.circle(frames, (x_temp,y_temp), radius=0, color=(0, 255, 0), thickness=4)
                                    #cv2.rectangle(frames, (x_temp - size_crop, y_temp - size_crop), (x_temp + size_crop, y_temp + size_crop), (255,0,0), 1)
                            #if (y_temp + size) <= heightf and (x_temp + size) <= widthf:
                cv2.drawContours(frames, contours, -1, (0,255,0), -1)

                # Assuming x_list and y_list have been populated as per your code
                points = list(zip(x_list, y_list))
                points_array = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
                # Now, points_array is in the correct format to be used as a contour
                # You can use this contour as needed, for example:
                # Draw the combined contour on the frame

                # Optionally, find the area or fit an ellipse
                ((centxo,centyo), (widtho,heighto), angleo) = cv2.fitEllipse(points_array)
                
                #Extracting the x and y values from the cap contour
                contourx = []
                contoury = []
                for ll in range(len(points_array)):
                    contourx.append(points_array[ll][0][0])
                    contoury.append(points_array[ll][0][1])
                    #cv2.circle(frame, (contourx[ii],contoury[ii]), radius=0, color=(255, 0, 0), thickness=-1)
                x_good = []
                y_good = []
                dis_good = []

                #cv2.drawContours(frames, contours, -1, (255,255,0), -1)
                #Filteringg the contour points 
                for dd in range(len(contourx)):
                    dis = math.hypot(contourx[dd] - centxo, contoury[dd] - centyo)
                    cv2.circle(frames, (contourx[dd],contoury[dd]), radius=0, color=(0, 0, 255), thickness=2)
                    diff_major = dis - heighto/2
                    if diff_major >= -(heighto/2) *.25:
                        x_good.append(contourx[dd])
                        y_good.append(contoury[dd])
                        dis_good.append(math.hypot(contourx[dd] - widthf/2, contoury[dd] - heightf/2))
                        #cv2.circle(frames, (x_list[dd],y_list[dd]), radius=0, color=(0, 255, 0), thickness=2)
                    #else:
                        #cv2.circle(frames, (x_list[dd],y_list[dd]), radius=0, color=(0, 0, 255), thickness=2)

                # Assuming x_list and y_list have been populated as per your code
                L = [[] for i in range(len(x_good))]
                for dd in range(len(x_good)):
                    #cv2.circle(frame, (x_good[dd],y_good[dd]), radius=0, color=(0, 0, 0), thickness=5)
                    L[dd] = [x_good[dd],y_good[dd]]
                ctr = np.array(L).reshape((-1,1,2)).astype(np.int32)
                #points_good = list(zip(x_good, y_good))
                #points_array_good = np.array(points_good, dtype=np.int32).reshape((-1, 1, 2))
                #print(len(points_array_good))
                #cv2.drawContours(frames, points_array_good, -1, (0,255,0), -1)
                # Now, points_array is in the correct format to be used as a contour
                # You can use this contour as needed, for example:
                # Draw the combined contour on the frame
                ((centx,centy), (width,height), angle) = cv2.fitEllipse(ctr)
                save_file_dir = f'/home/pruthvi/Desktop/MARCI_VIDEOS/CODE/SOUTH_POLE/OUTLINES/MY_{28+jj}'
                filename = os.path.basename(image_list[ii])
                full_path = f'{save_file_dir}/{filename}'
                print(full_path)
                cv2.imwrite(full_path, frames)  # Assuming 'frames' is your image data
