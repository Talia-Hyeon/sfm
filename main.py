import sys
import os
from os import path as osp

import cv2
import numpy as np
import matplotlib.pyplot as plt

# intrinsic matrix
img_root = '../data/'
txt_file = osp.join(img_root, 'K.txt')
K = np.loadtxt(txt_file)


def load_image(path):
    img_file = [osp.join(path, file) for file in os.listdir(path) if file.endswith('.JPG')]
    # for i in len(img_file) - 1:
    #     img1 = img_file[i]
    #     img2 = img_file[i + 1]
    #     # load, extract feature, ..., triangulation
    img1 = img_file[0]
    img2 = img_file[1]
    img1 = cv2.imread(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.imread(img2, cv2.COLOR_BGR2RGB)  # IMREAD_GRAYSCALE
    return img1, img2


def extract_feature(img1, img2):
    feature = cv2.KAZE_create()
    # feature = cv2.AKAZE_create()
    # feature = cv2.ORB_create()
    kp1, desc1 = feature.detectAndCompute(img1, None)
    kp2, desc2 = feature.detectAndCompute(img2, None)

    # draw keypoints
    img_draw = cv2.drawKeypoints(img1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(img_draw)
    plt.show()
    return kp1, desc1, kp2, desc2


def match_keypoints(img1, kp1, desc1, img2, kp2, desc2):
    matcher = cv2.BFMatcher_create()
    matches = matcher.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)  # len(matches)  # # of matching point
    good_matches = matches[:80]  # 좋은 매칭 결과 80개

    img_match = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, img2, flags=2)
    plt.imshow(img_match)
    plt.show()
    return good_matches


def essentialMat_estimation(kp1, kp2, good_matches, K):
    # matches의 type(dematch): queryIndex + trainIndex
    # queryIndex: 1번 img keypoint 번호
    # trainIndex: 2번 img keypoint 번호
    pts1 = np.array([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2).astype(np.float32)  # 픽셀 좌표
    pts2 = np.array([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2).astype(np.float32)
    # reshape(len(good_matches),1,2)
    E, mask = cv2.findEssentialMat(pts1, pts2, cameraMatrix=K, method=cv2.RANSAC)

    # essential matrix decomposition
    # R1, R2, t = cv2.decomposeEssentialMat(E)
    retval, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)  # pts, K...?
    return R, t


def triangulate(R, t, K, p1, p2):
    Rt0 = np.hstack((np.eye(3), np.zeros((3, 1))))
    Rt1 = np.hstack((R, t))
    Rt1 = np.matmul(K, Rt1)

    pt1 = np.transpose(p1)
    pt2 = np.transpose(p2)

    p3d = cv2.triangulatePoints(Rt0, Rt1, pt1, pt2)
    p3d /= p3d[3]  # Homogeneous Coordinate
    return p3d



if __name__ == '__main__':
    img1, img2 = load_image(img_root)
    kp1, desc1, kp2, desc2 = extract_feature(img1, img2)
    good_matches = match_keypoints(img1, kp1, desc1, img2, kp2, desc2)
    E = essentialMat_estimation(kp1, kp2, good_matches, K)
