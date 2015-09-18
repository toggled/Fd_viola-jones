import numpy as np
from HaarLikeFeature import FeatureType
from HaarLikeFeature import HaarLikeFeature
from HaarLikeFeature import FeatureTypes
import sys

windowx = 25
windowy = 25
class AdaBoost(object):
    '''
    classdocs
    '''


    def __init__(self, params):
        '''
        Constructor
        '''
    
def learn(positives, negatives, T):
    
    # construct initial weights
    pos_weight = 1. / (2 * len(positives))
    neg_weight = 1. / (2 * len(negatives))
    for p in positives:
        p.set_weight(pos_weight)
    for n in negatives:
        n.set_weight(neg_weight)
    
    # create column vector
    images = np.hstack((positives, negatives))
    
    print 'Creating haar like features..'
    features = []
    for f in FeatureTypes:
        for width in range(f[0],windowx,f[0]):
            for height in range(f[1],windowy,f[1]):
                for x in range(windowx-width):
                    for y in range(windowy-height):
                        features.append(HaarLikeFeature(f, (x,y), width, height, 0, 1))
    print '..done.\n' + str(len(features)) + ' features created.\n'
    

    
    print 'Calculating scores for features..'
    # dictionary of feature -> list of vote for each image: matrix[image, weight, vote])
    votes = dict()
    i = 0
    for feature in features:
        # calculate score for each image, also associate the image
        feature_votes = np.array(map(lambda im: [im, feature.get_vote(im)], images))
        votes[feature] = feature_votes
        i += 1
       # if i % 10 == 0:
            #break   #@todo: remove
          #  print str(i) + ' features of ' + str(len(features)) + ' done'
    print '..done.\n'
    

    
    # select classifiers
    
    classifiers = []
    used = []
    
    print 'Selecting classifiers..'
    sys.stdout.write('[' + ' '*20 + ']\r')
    sys.stdout.flush()
    for i in range(T):
        print 'iteration: ',i
        classification_errors = dict()

        # normalize weights
        for im in images:
            print im.weight
         
        norm_factor = 1. / sum(map(lambda im: im.weight, images))
        for image in images:
            image.set_weight(image.weight * norm_factor)
        if(len(used)==len(votes)):
            break
        x = 1
        # select best weak classifier
        for feature, feature_votes in votes.iteritems():
            print "feature number: ",x, feature.type,'position: ',feature.top_left
            x+=1
            if feature in used:
                #print 'continue'
                continue
                
            #for each feature set its threshold
                #find threshold
            #print feature
           
            integralimage_sorted = np.array(sorted(feature.score.items(),key = lambda x: x[1]))[:,0]
            
            #print integralimage_sorted
            #print feature_sorted_label
            
            #for intimg,featurelabel in sorted(feature.score.items(),key = lambda x: x[1]):
             #   print intimg.weight,'-> ',intimg.label,' -> ',feature.get_score(intimg)," "
            
            #find optimal parameters for each decision_stump/feature
            feature.threshold = min(feature.score.values()) - 1
            print 'threshold: ',feature.threshold
            w1gt_pos = sum(map(lambda im,featureval:im.weight if feature.get_score(im)>feature.threshold and im.label==1 else 0, feature_votes[:,0], feature_votes[:,1]))
            # sum of pos examples whose feature value is > threshold
            #print "w1+_pos: ",w1gt_pos
            
            w1gt_neg = sum(map(lambda im,featureval:im.weight if feature.get_score(im)>feature.threshold and im.label ==-1 else 0, feature_votes[:,0],feature_votes[:,1]))
            # sum of neg examples whose feature value is > threshold
            
            #print "w1+_neg ",w1gt_neg
            w1lt_pos = 0 #sum of pos examples whose feature value is < threshold
            w1lt_neg = 0 #sum of neg examples whose feature value is < threshold 
            
            j = -1
            margin_prime = feature.margin
            tao_prime = feature.threshold
            while 1:
                error_pluspolarity = w1gt_pos + w1lt_neg  
                error_minuspolarity = w1gt_neg + w1lt_pos
                                
            #    print 'Margin:',margin_prime
                
                #print 'now', error_pluspolarity,' ',error_minuspolarity
                #for im,featureval in zip(feature_votes[:,0],feature_votes[:,1]):
                    #print abs(feature.get_score(im)-feature.threshold)
                
                if error_pluspolarity > error_minuspolarity:
                    epsilon_prime = error_minuspolarity
                    polarity_prime = -1
                else:
                    epsilon_prime = error_pluspolarity 
                    polarity_prime = 1 # capital Tao 
                    #margin_prime = min(map(lambda im,featureval: abs(feature.get_score(im)-feature.threshold) , feature_votes[:,0], feature_votes[:,1]))
                #print "epsilon prime: ",epsilon_prime," feature.epsilon: ",feature.Epsilon
                if epsilon_prime < feature.Epsilon or epsilon_prime == feature.Epsilon and margin_prime>feature.margin:
                    feature.Epsilon = epsilon_prime
                    feature.polarity = polarity_prime
                    feature.margin = margin_prime
                    feature.threshold = tao_prime
                    #print 'updating feature epsilon,polarity , margin and threshold'
                    
                if j == len(images)-1:
                     break
                j+=1
                
                #eliminating duplicates
                while 1:
                    #print j,' '
                    
                    if integralimage_sorted[j].label == -1: #negative example
                        #print integralimage_sorted[j].imgsrc
                        
                        w1lt_neg += integralimage_sorted[j].weight
                        w1gt_neg -= integralimage_sorted[j].weight
                    else: #positive example
                       
                        w1gt_pos -= integralimage_sorted[j].weight
                        w1lt_pos += integralimage_sorted[j].weight
                    
                    if j == len(images)-1 or feature.score[integralimage_sorted[j]] != feature.score[integralimage_sorted[j+1]]:
                        break
                    else:
                        j+=1
                if j == len(images)-1:
                    tao_prime = max(feature.score.values()) + 1
                    margin_prime = 0
                    #print 'j increased to ',len(images)-1
                else:
                    tao_prime = (feature.score[integralimage_sorted[j]]+feature.score[integralimage_sorted[j+1]])/2
                    margin_prime = feature.score[integralimage_sorted[j+1]] - feature.score[integralimage_sorted[j]]
                    #print 'threshold changed: ',tao_prime
                    #print 'margin_prime changed: ',margin_prime
                    
                    
                    
            #log the features by their missclassification error. If two feat has same error log the one with wider margin
            
            feat_votes_afteroptimization = np.array(map(lambda im: [im, feature.get_vote(im)], images))
            # calculate weighted error
            error = sum(map(lambda im, vote: im.weight if im.label != vote else 0, feat_votes_afteroptimization[:,0], feat_votes_afteroptimization[:,1]))
           
            # map error -> feature, use error as key to select feature with
            print 'error: ',error
            #print feature.threshold,' ',feature.polarity
            
            if classification_errors.get(error,None) : #there is another feature logged previosly with the same error 
                if feature.margin > classification_errors[error].margin: #update it only if it has lower margin 
                    classification_errors[error] = feature
            
            else:
                classification_errors[error] = feature
            print 'properties: ',feature.threshold,' ',feature.polarity,' ',feature.margin,' ',feature.Epsilon
            
        #print classification_errors
        
        # get best feature, i.e. with smallest weighted misclassification error
        errors = classification_errors.keys()
        
            
        best_error = errors[np.argmin(errors)]
        feature = classification_errors[best_error]
        print best_error
        
        if best_error != 0:
            used.append(feature)
            feature_weight = 0.5 * np.log((1-best_error)/best_error)
            classifiers.append((feature, feature_weight))
        else:  
            classifiers.append((feature, 1))
            continue
        
        # update image weights
        best_feature_votes = votes[feature]
        for feature_vote in best_feature_votes:
            im = feature_vote[0]
            vote = feature_vote[1]
            if im.label != vote:
                im.set_weight(im.weight * np.sqrt((1-best_error)/best_error))
            else:
                im.set_weight(im.weight * np.sqrt(best_error/(1-best_error)))
        
        sys.stdout.write('[' + '='*(((i+1)*20)/T) + ' '*(20-(((i+1)*20)/T)) + ']\r')
        sys.stdout.flush()
    print '..done.\n'
    
    return classifiers
        