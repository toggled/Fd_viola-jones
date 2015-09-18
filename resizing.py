import  os, sys

import operator
from PIL import Image
# loads all the gif of jpg images (must be histogram equalized) in the path folder and converts them into 24x24 size
path = 'training/nonfaces'
size = (24,24)

def load_images(path):
    images = []
    for _file in os.listdir(path):
        #print _file
        if _file.endswith('.jpg') or _file.endswith('.gif') or _file.endswith('.png'):
            temp = Image.open(os.path.join(path, _file))
            images.append( (temp,_file) )
    return images
    
imagelist = load_images(path)


for image,filename in imagelist: 
    #gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray_histeq_image = image
    gray_histeq_image.thumbnail(size)
    
    
    #gray_image.show()
    #equ.show()
    gray_histeq_image.save(os.path.join(path,filename))