import  os, sys

import operator
from PIL import Image
# loads all the gif of jpg images (currently must be gray scale) in the path folder and converts them into histogram equalized image. 
path = 'training/faces/test'

def equalize(h):

    lut = []

    for b in range(0, len(h), 256):

        # step size
        step = reduce(operator.add, h[b:b+256]) / 255

        # create equalization lookup table
        n = 0
        for i in range(256):
            lut.append(n / step)
            n = n + h[i+b]

    return lut
    
def load_images(path):
    images = []
    for _file in os.listdir(path):
        #print _file
        if _file.endswith('.jpg') or _file.endswith('.gif'):
            temp = Image.open(os.path.join(path, _file))
            images.append( (temp,_file) )
    return images
    
imagelist = load_images(path)


for image,filename in imagelist: 
    #gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray_image = image
    
    equ = gray_image.point( equalize(gray_image.histogram()) )
    #gray_image.show()
    #equ.show()
    equ.save(os.path.join(path,filename))