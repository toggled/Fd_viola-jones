import AdaBoost
from IntegralImage import IntegralImage
import os,sys

    
def load_images(path, label):
    images = []
    for _file in os.listdir(path):
        if _file.endswith('.jpg') or _file.endswith('.gif'):
            images.append(IntegralImage(os.path.join(path, _file), label))
    return images

def classify(classifiers, image):
    sum = 0
    for c in classifiers:
        # c is a weak classifier. c[0] is the important feature and c[1] its weight
        if c[0].get_vote(image)>0:
            c[0].drawbox(image)
           
        sum+=c[0].get_vote(image)*c[1]
    if sum>0:
        return 1
    else:
        return -1
    
        

if __name__ == "__main__":
    
    # TODO: select optimal threshold for each feature
    # TODO: attentional cascading
    
    print 'Loading faces..'
    faces = load_images('training/faces', 1)
    print '..done. ' + str(len(faces)) + ' faces loaded.\n\nLoading non faces..'
    non_faces = load_images('training/nonfaces', -1)
    print '..done. ' + str(len(non_faces)) + ' non faces loaded.\n'
    
    T = 20
    classifiers = AdaBoost.learn(faces, non_faces, T)
    
    print 'Loading test faces..'
    faces = load_images('training/faces/test', 1)
    print '..done. ' + str(len(faces)) + ' faces loaded.\n\nLoading test non faces..'
    non_faces = load_images('training/nonfaces/test', -1)
    print '..done. ' + str(len(non_faces)) + ' non faces loaded.\n'
    
    print 'Validating selected classifiers..'
    correct_faces = 0
    correct_non_faces = 0
    for image in faces:
        result = classify(classifiers, image)
        if image.label == 1 and result == 1:
            #image.originalimage.show()
            image.show()
            correct_faces += 1
        if image.label == -1 and result == -1:
            correct_non_faces += 1
            #image.originalimage.show()
            
    print '..done. Result:\n  Faces: ' + str(correct_faces) + '/' + str(len(faces)) + '\n  non-Faces: ' + str(correct_non_faces) + '/' + str(len(non_faces))
