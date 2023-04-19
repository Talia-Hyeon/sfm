import cv2
import numpy as np
import sys
import os
from os import path as osp

# intrinsic matrix
img_root = '../data/'
txt_file = osp.join(img_root, 'K.txt')
K = np.loadtxt(txt_file)

# load image
img_file = [osp.join(img_root, file) for file in os.listdir(img_root) if file.endswith('.JPG')]
# for img in img_file:
img1 = img_file[0]
img2 = img_file[1]
img1 = cv2.imread(img1, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img2, cv2.IMREAD_GRAYSCALE)

# compute keypoints and descriptors
feature = cv2.KAZE_create()
# feature = cv2.AKAZE_create()
# feature = cv2.ORB_create()
kp1, desc1 = feature.detectAndCompute(img1, None)
kp2, desc2 = feature.detectAndCompute(img2, None)

# print('keypoint:', len(kp1), 'descriptor:', desc1.shape)
# # of keypoint, descriptor: keypoint 1개 당 64개 feature descriptor 값 사용
# # draw keypoints
# img_draw = cv2.drawKeypoints(img1, kp1, None,
#                              flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# cv2.imshow('KAZE', img_draw)
# cv2.waitKey(3000)  # args=0: 입력 있을 때까지 대기
# cv2.destroyAllWindows()  # 창 모두 닫기

# match keypoints
matcher = cv2.BFMatcher_create()
matches = matcher.match(desc1, desc2)
matches = sorted(matches, key=lambda x: x.distance)
# print(len(matches))  # # of matching point
good_matches = matches[:80]  # 좋은 매칭 결과 5개

# essential matrix estimation
pts1 = np.array([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2).astype(np.float32)
pts2 = np.array([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2).astype(np.float32)
# matches의 dematch type에는 queryIndex: 1번 img keypoint 번호, trainIndex
E = cv2.findEssentialMat(pts1, pts2, cameraMatrix=K, method=cv2.RANSAC)

# 호모그래피를 이용하여 기준 영상 영역 표시
H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC)
dst = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None,
                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
(h, w) = img1.shape[:2]
corners1 = np.array([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2).astype(np.float32)
corners2 = cv2.perspectiveTransform(corners1, H)
corners2 = corners2 + np.float32([w, 0])
cv2.polylines(dst, [np.int32(corners2)], True, (0, 255, 0), 2, cv2.LINE_AA)
cv2.namedWindow('dst', cv2.WINDOW_NORMAL)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
