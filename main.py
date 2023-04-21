import os
from os import path as osp

import cv2
import numpy as np
import matplotlib.pyplot as plt


def read_files(path):
    txt_file = osp.join(path, 'K.txt')
    K = np.loadtxt(txt_file)
    img_file = [osp.join(path, file) for file in os.listdir(path) if file.endswith('.JPG')]
    return K, img_file


def extract_feature(img1, img2):
    feature = cv2.KAZE_create()
    # feature = cv2.AKAZE_create()
    # feature = cv2.ORB_create()
    kp1, desc1 = feature.detectAndCompute(img1, None)
    kp2, desc2 = feature.detectAndCompute(img2, None)

    # draw keypoints
    img_draw = cv2.drawKeypoints(img2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(img_draw)
    plt.show(block=False)
    plt.pause(2)
    plt.close()
    return kp1, desc1, kp2, desc2


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


def init_projMat(K, R, t):
    Rt0 = np.hstack((np.eye(3), np.zeros((3, 1))))  # 1st camera coordinate: world coordinate
    Rt1 = np.hstack((R, t))
    Rt1 = np.matmul(K, Rt1)
    return Rt0, Rt1


def triangulate(R, t, K, p1, p2):
    Rt0 = np.hstack((np.eye(3), np.zeros((3, 1))))  # 1st camera coordinate: world coordinate
    Rt1 = np.hstack((R, t))
    Rt1 = np.matmul(K, Rt1)

    pt1 = np.transpose(p1)
    pt2 = np.transpose(p2)

    p3d = cv2.triangulatePoints(Rt0, Rt1, pt1, pt2)
    p3d /= p3d[3]  # Homogeneous Coordinate
    # p3d's shape: (4,1,80)  4:x,y,z +?
    return p3d


# Rescale to Homogeneous Coordinate
def rescale_point(pts1, pts2, length):
    p1 = [[]]
    p2 = [[]]
    for i in range(length):
        tmp1 = pts1[i].flatten()
        tmp1 = np.append(tmp1, 1)
        p1 = np.append(p1, tmp1)
        tmp2 = pts2[i].flatten()
        tmp2 = np.append(tmp2, 1)
        p2 = np.append(p2, tmp2)

    p1 = p1.reshape((length), 3)
    p2 = p2.reshape((length), 3)
    return p1, p2


# Triangulation
def LinearTriangulation(Rt0, Rt1, p1, p2):
    A = [p1[1] * Rt0[2, :] - Rt0[1, :],  # x(p 3row) - (p 1row)
         -(p1[0] * Rt0[2, :] - Rt0[0, :]),  # y(p 3row) - (p 2row)
         p2[1] * Rt1[2, :] - Rt1[1, :],  # x'(p' 3row) - (p' 1row)
         -(p2[0] * Rt1[2, :] - Rt1[0, :])]  # y'(p' 3row) - (p' 2row)

    A = np.array(A).reshape((4, 4))
    AA = A.T @ A
    U, S, VT = np.linalg.svd(AA)  # right singular vector

    return VT[3, 0:3] / VT[3, 3]


def make_3dpoint(Rt0, Rt1, p1, p2):
    p3ds = []
    for pt1, pt2 in zip(p1, p2):
        p3d = LinearTriangulation(Rt0, Rt1, pt1, pt2)
        # p3d.shape: (3,)
        p3ds.append(p3d)
    p3ds = np.array(p3ds).T
    return p3ds


def reconstruct(K, img1, img2):
    kp1, desc1, kp2, desc2 = extract_feature(img1, img2)
    good_matches = match_keypoints(img1, kp1, desc1, img2, kp2, desc2)
    R, t, pts1, pts2 = essentialMat_estimation(kp1, kp2, good_matches, K)
    # point_3d = triangulate(R, t, K, pts1, pts2)
    Rt0, Rt1 = init_projMat(K, R, t)
    p1, p2 = rescale_point(pts1, pts2, len(pts1))
    point_3d = make_3dpoint(Rt0, Rt1, p1, p2)
    return point_3d


def visualization(p3ds):
    X = np.array([])
    Y = np.array([])
    Z = np.array([])  # 120
    X = np.concatenate((X, p3ds[0]))
    Y = np.concatenate((Y, p3ds[1]))
    Z = np.concatenate((Z, p3ds[2]))

    fig = plt.figure(figsize=(15, 15))
    ax = plt.axes(projection='3d')
    ax.scatter3D(X, Y, Z, c='b', marker='o')
    plt.show()


def run(path):
    K, img_file = read_files(img_root)

    # baseline
    img1 = img_file[0]
    img2 = img_file[1]
    img1 = cv2.imread(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.imread(img2, cv2.COLOR_BGR2RGB)
    point_3d = reconstruct(K, img1, img2)

    # not baseline
    for i in range(2, len(img_file)):
        img2 = img_file[i]
        img2 = cv2.imread(img2, cv2.COLOR_BGR2RGB)
        point_3d_tem = reconstruct(K, img1, img2)
        point_3d = np.hstack((point_3d, point_3d_tem))
        print("point_3d's shape: {}".format(point_3d.shape))

    visualization(point_3d)


if __name__ == '__main__':
    img_root = '../data/'
    run(img_root)
    # test
