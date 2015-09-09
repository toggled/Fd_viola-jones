import cv2
import os

for i in os.listdir('.'):
    if 'png' in i or 'jpg' in i:
        image = cv2.imread(i)
        print image.shape
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            cv2.imwrite(i,gray_image)
#print gray_image[toplefty:bottrighty,topleftx:bottrightx]

#cv2.rectangle(gray_image,(topleftx,toplefty),(bottrightx,bottrighty),(0,0,0),1)
#cv2.imshow('color_image',image)
#cv2.imshow('gray_image',gray_image)
#for i in range(bottrightx-topleftx+1):
    #for j in range(bottrighty-toplefty+1):

#print gray_image[1]
   # print '\n'


raw_input()