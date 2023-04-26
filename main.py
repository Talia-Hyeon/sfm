import os
from os import path as osp

import cv2
import numpy as np
import matplotlib.pyplot as plt


def read_files(path):
    txt_file = osp.join(path, 'K.txt')
    K = np.loadtxt(txt_file)
    img_filess = [osp.join(path, file) for file in os.listdir(path) if file.endswith('.JPG')]
    return K, img_filess


def extract_feature(img):
    feature = cv2.KAZE_create()
    # feature = cv2.AKAZE_create()
    # feature = cv2.ORB_create()
    kp, desc = feature.detectAndCompute(img, None)

    return kp, desc


def create_view(self, img_files):
    view_l = []
    for i in img_files:
        i_img = cv2.imread(i, cv2.COLOR_BGR2RGB)
        kp, desc = extract_feature(i_img)
        view = {['img_path']: i, ['kp']: kp, ['desc']: desc}
        view_l.append(view)

    # draw keypoints
    img = cv2.imread(view_l[0]['img_path'], cv2.COLOR_BGR2RGB)
    kp = view_l[0]['kp']
    img_draw = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(img_draw)
    plt.show()
    return view_l


def match_keypoints(img1, kp1, desc1, img2, kp2, desc2):
    matcher = cv2.BFMatcher_create()
    matches = matcher.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)  # len(matches)  # # of matching point
    good_matches = matches[:80]  # 좋은 매칭 결과 80개

    img_match = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, img2, flags=2)
    plt.imshow(img_match)
    plt.show(block=False)
    plt.pause(2)
    plt.close()
    return good_matches


def essentialMat_estimation(kp1, kp2, good_matches, K):
    # matches의 class(dematch)의 attribution: queryIndex + trainIndex
    # queryIndex: 1번 img keypoint 번호
    # trainIndex: 2번 img keypoint 번호
    pts1 = np.array([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2).astype(np.float32)  # 픽셀 좌표
    pts2 = np.array([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2).astype(np.float32)
    # reshape(len(good_matches),1,2)
    # pts'shape:(80, 1, 2) -> # of matches, x&y 좌표
    E, mask = cv2.findEssentialMat(pts1, pts2, cameraMatrix=K, method=cv2.RANSAC)
    pts1 = pts1[mask.ravel() == 1]  # img 1 inlier
    pts2 = pts2[mask.ravel() == 1]  # img 1 inlier

    # essential matrix decomposition
    retval, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
    # R1, R2, t = cv2.decomposeEssentialMat(E)
    return R, t, pts1, pts2


def triangulate(R, t, K, p1, p2):
    Rt0 = np.hstack((np.eye(3), np.zeros((3, 1))))  # 1st camera coordinate: world coordinate
    Rt1 = np.hstack((R, t))
    Rt1 = np.matmul(K, Rt1)

    pt1 = np.transpose(p1)
    pt2 = np.transpose(p2)

    p3d = cv2.triangulatePoints(Rt0, Rt1, pt1, pt2)
    p3d /= p3d[3]  # Homogeneous Coordinate
    # p3d's shape: (4,1,80)  4:x,y,z,1
    return p3d


def reconstruct(K, img1, img2):
    kp1, desc1 = extract_feature(img1)
    kp2, desc2 = extract_feature(img2)
    good_matches = match_keypoints(img1, kp1, desc1, img2, kp2, desc2)
    R, t, pts1, pts2 = essentialMat_estimation(kp1, kp2, good_matches, K)
    point_3d = triangulate(R, t, K, pts1, pts2)
    return point_3d


def run(path):
    K, img_files = read_files(img_root)

    # baseline
    img1 = img_files[0]
    img2 = img_files[1]
    img1 = cv2.imread(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.imread(img2, cv2.COLOR_BGR2RGB)
    point_3d = reconstruct(K, img1, img2)

    # not baseline
    for i in range(2, len(img_files)):
        img2 = img_files[i]
        img2 = cv2.imread(img2, cv2.COLOR_BGR2RGB)
        point_3d_tem = reconstruct(K, img1, img2)
        point_3d = np.hstack((point_3d, point_3d_tem))
        print("point_3d's shape: {}".format(point_3d.shape))


if __name__ == '__main__':
    img_root = '../data/'
    run(img_root)
