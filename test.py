import cv2
import orb_matcher
import numpy as np
import orb
OE = orb.orb_ORBextractor(2048,1.2,8,20,5)
i = 10
im1 = cv2.imread('/home/olorin/Downloads/IISC_Road_data/cds_dept/1.bag/left_image/frame0%03d.jpg'%(i), 0)
im2 = cv2.imread('/home/olorin/Downloads/IISC_Road_data/cds_dept/1.bag/left_image/frame0%03d.jpg'%(i+1), 0)
EM = orb_matcher.orb_matcher_ORBmatcher(0.72)


x1,y1 = OE.extract_orb_fts(im1, None)
x2,y2 = OE.extract_orb_fts(im2, None)
no,matches = EM.find_matches(x1,x2,y1,y2,100)

cv_kp1 = [cv2.KeyPoint(x=pt[0], y=pt[1], _size=1) for pt in x1]
cv_kp2 = [cv2.KeyPoint(x=pt[0], y=pt[1], _size=1) for pt in x2]

matches = matches.T

dmatches = []
for match1, match2 in enumerate(matches):
    if match2!=-1:
        dmatches.append(cv2.DMatch(_imgIdx=0,_queryIdx=match1, _trainIdx=match2, _distance = 0))

print("FILE", i,i+1, len(dmatches), no)
out_img = cv2.drawMatches(im1, cv_kp1, im2, cv_kp2, dmatches, None, -1, -1, None, 2)
cv2.imshow("IM", out_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

