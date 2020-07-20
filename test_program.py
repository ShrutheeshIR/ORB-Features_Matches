import cv2
import orb_matcher
import numpy as np
import orb
import time
OE = orb.orb_ORBextractor(2048,1.2,8,20,5)
cammat = np.array([[349.9, 0, 335.3375], [0, 349.9, 199.2915], [0,0,1]])
bigarr = []
for i in range(0,1279,1):
    # print()
    im1 = cv2.imread('/home/olorin/Downloads/IISC_Road_data/cds_dept/1.bag/left_image/frame%04d.jpg'%(i), 0)
    im2 = cv2.imread('/home/olorin/Downloads/IISC_Road_data/cds_dept/1.bag/left_image/frame%04d.jpg'%(i+1), 0)
    EM = orb_matcher.orb_matcher_ORBmatcher(372,376,0.72)
    

    x1,y1 = OE.extract_orb_fts(im1, None)
    x2,y2 = OE.extract_orb_fts(im2, None)
    print(x2.shape, x1.shape)
    no,matches = EM.find_matches(x1,x2,y1,y2,100, 2)
    # for val in x1:
    #     if val[0]>95 and val[0]<97:
    #         print(val)
    # print(x1)

    cv_kp1 = [cv2.KeyPoint(x=pt[0], y=pt[1], _size=1) for pt in x1]
    cv_kp2 = [cv2.KeyPoint(x=pt[0], y=pt[1], _size=1) for pt in x2]

    # print("@@@@@@",len(matches[matches!=-1]))
    matches = matches.T

    dmatches = []
    pos_a = []
    pos_b = []
    for match1, match2 in enumerate(matches):
        if match2!=-1:
            dmatches.append(cv2.DMatch(_imgIdx=0,_queryIdx=match1, _trainIdx=match2[0], _distance = 0))
            pos_a.append([x1[match1][0], x1[match1][1]])
            # print(match1, match2)
            pos_b.append([x2[match2[0]][0], x2[match2[0]][1]])


    pos_a = np.array(pos_a)
    pos_b = np.array(pos_b)
    # print(type(pos_a), pos_a.shape, type(pos_b), pos_b.shape )

    E, inliers =cv2.findEssentialMat(
                pos_b,
                pos_a,
                cammat,
                method=cv2.RANSAC,
                prob=0.99,
                threshold=1,
                )
    # t3= time.time()
    inliers = np.squeeze(inliers)
    inliers1 = inliers.copy()
    points, R, t, _ = cv2.recoverPose(E, pos_b, pos_a, mask = inliers1)
    print("FILE", i,i+1, len(dmatches), points)
    bigarr.append(points)
    out_img = cv2.drawMatches(im1, cv_kp1, im2, cv_kp2, dmatches, None, -1, -1, None, 2)
    cv2.imshow("IM", out_img)
    cv2.waitKey(1)
cv2.destroyAllWindows()
bigarr = np.array(bigarr)
np.save('bigarr-orb.npy', bigarr)
# print(dmatches)
# for(int i = 0; i<matches12.size();i++)
# {
#     // std::cout<<matches12[i]<<"_";
#     if(matches12[i]!=65535)
#     {
#         cv2.DMatch dm(i, matches12[i], 1);
#         dmatches.push_back(dm);
#     }
# }
