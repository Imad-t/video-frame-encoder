import cv2
import numpy as np
from math import inf
import time



img1 = cv2.imread('image1.png')
img2 = cv2.imread('image2.png')

#transform images to ycrcb to separate chrominance and luminance and then eliminate chrominance
grayImg1 = cv2.cvtColor(img1, cv2.COLOR_BGR2YCrCb)[:,:,0]
grayImg2 = cv2.cvtColor(img2, cv2.COLOR_BGR2YCrCb)[:,:,0]

#calculate the mean square error between two blocks
def Mse(block1, block2):
   err = np.sum((block1- block2)**2)
   err = err / (block1.shape[0]*block1.shape[1])
   return err

#processing

blocs1 =[]
blocs2 =[]
predicted_img = (np.ones(grayImg1.shape)*255).astype(np.uint8)
search_dist = 7

start = time.time()

#loop through img1 blocks
for i in range(0, grayImg1.shape[0]-16, 16):
    for j in range(0, grayImg1.shape[1]-16, 16):

        block1 = grayImg1[i:(i+16), (j):(j+16)]
        current_block = [i,(i+16),j,(j+16)]
        min_diff = inf
        #loop through img2 blocks and compare with current block
        for m in range(max((i-search_dist),0), min((i+search_dist),(grayImg1.shape[0]-15))):
            for n in range(max((j-search_dist),0), min((j+search_dist),(grayImg1.shape[1]-15))):
                block2 = grayImg2[m:(m+16), n:(n+16)]
                diff = Mse(block1, block2)
                if diff < min_diff :
                    min_diff = diff 
                    tmp_block = [m,m+(16),n,n+(16)]
        if min_diff>=50:
            img1 = cv2.rectangle(img1, (current_block[2], current_block[0]), (current_block[3], current_block[1]), (0, 255, 0),2)
            img2 = img2 = cv2.rectangle(img2, (tmp_block[2], tmp_block[0]), (tmp_block[3], tmp_block[1]), (255, 0, 0), 2)

        #replace current block with the most similar block 
        predicted_img[i:(i+16), j:(j+16)]  = grayImg2[tmp_block[0]:tmp_block[1], tmp_block[2]:tmp_block[3]]

resedu = grayImg1 - predicted_img

#construct image from predicted image and resedu
constructed = resedu + predicted_img

print("Execution time: " + str(time.time()-start) + " s")

#dispalay results

#show blocks with high mse
cv2.imshow("image1", img1)
cv2.waitKey(1)
cv2.imshow("image2", img2)
cv2.waitKey(1)

cv2.imshow("predicted image", predicted_img)
cv2.waitKey(0)

cv2.imshow("resedu image", resedu)
cv2.waitKey(0)

cv2.imshow("constructed image", constructed)
cv2.waitKey(0)

cv2.waitKey(0)
cv2.destroyAllWindows()
