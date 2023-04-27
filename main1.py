import os
from os import path as osp

import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d


def read_files(path):
    txt_file = osp.join(path, 'K.txt')
    K = np.loadtxt(txt_file)
    # files=[for file in os.listdir(path) if file.endswith('.JPG')]
    img_files = [osp.join(path, file) for file in os.listdir(path) if file.endswith('.JPG')]
    img_files.sort()
    return K, img_files


def extract_feature(img):
    feature = cv2.SIFT_create()
    # feature = cv2.KAZE_create()
    # feature = cv2.AKAZE_create()
    # feature = cv2.ORB_create()
    kp, desc = feature.detectAndCompute(img, None)
    return kp, desc


def create_view(img_files):
    view_l = []
    for i in img_files:
        i_img = cv2.imread(i, cv2.COLOR_BGR2RGB)
        kp, desc = extract_feature(i_img)
        view = {'img': i_img, 'img_path': i, 'kp': kp, 'desc': desc}
        view_l.append(view)

    # draw keypoints
    img = view_l[0]['img']
    kp = view_l[0]['kp']
    img_draw = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(img_draw)
    # plt.show()
    return view_l


def match_keypoints(view1, view2):
    matcher = cv2.BFMatcher_create()
    matches = matcher.knnMatch(view1['desc'], view2['desc'], k=2)
    good_matches = [m1 for m1, m2 in matches if m1.distance < 0.80 * m2.distance]  # 좋은 매칭 결과
    sort_matches = sorted(good_matches, key=lambda x: x.distance)  # len(matches)  # # of matching point

    # draw matches
    img_match = cv2.drawMatches(view1['img'], view1['kp'], view2['img'], view2['kp'],
                                sort_matches, view2['img'], flags=2)
    plt.imshow(img_match)
    plt.show()
    return sort_matches


def essentialMat_estimation(kp1, kp2, good_matches, K):
    # preprocessing
    pts1 = np.array([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2).astype(np.float32)  # 픽셀 좌표
    pts2 = np.array([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2).astype(np.float32)
    # matches의 class(dematch)의 attribution: queryIndex + trainIndex
    # queryIndex: 1번 img keypoint 번호
    # trainIndex: 2번 img keypoint 번호
    # reshape(len(good_matches),1,2)
    # pts'shape:(80, 1, 2) -> # of matches, x&y 좌표

    E, mask = cv2.findEssentialMat(pts1, pts2, cameraMatrix=K, method=cv2.RANSAC)

    # Removes points that have already been reconstructed in the completed views
    pts1 = pts1[mask.ravel() == 1]  # img 1 inlier
    pts2 = pts2[mask.ravel() == 1]  # img 2 inlier

    # essential matrix decomposition
    retval, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
    # R1, R2, t = cv2.decomposeEssentialMat(E)
    return R, t, pts1, pts2


def visualization(p3ds):
    X = np.array([])
    Y = np.array([])
    Z = np.array([])  # 120
    print("p3ds's shape: {}".format(p3ds.shape))
    X = np.concatenate((X, p3ds[:, 0]))
    Y = np.concatenate((Y, p3ds[:, 1]))
    Z = np.concatenate((Z, p3ds[:, 2]))

    # fig = plt.figure(figsize=(15, 15))
    ax = plt.axes(projection='3d')
    ax.scatter3D(X, Y, Z, c='b', marker='o')
    plt.show()


def triangulate(R, t, K, p1, p2):
    Rt0 = np.hstack((np.eye(3), np.zeros((3, 1))))  # 1st camera coordinate: world coordinate
    Rt0 = K @ Rt0
    Rt1 = np.hstack((R, t))
    Rt1 = np.matmul(K, Rt1)

    pt1 = np.transpose(p1)
    pt2 = np.transpose(p2)
    pt1 = np.squeeze(pt1)
    pt2 = np.squeeze(pt2)

    print("pt1'shape:{}".format(pt1.shape))

    p3d = cv2.triangulatePoints(Rt0, Rt1, pt1, pt2)
    p3d /= p3d[3]  # Homogeneous Coordinate: [[[x]] [[y]] [[z]] [[1]]]
    # p3d's shape: (4,1,80)
    p3d = np.squeeze(p3d)[:3]  # (4,80)
    return p3d


def compute_PNP(K, view, points_3D, done):
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # collects all the descriptors of the reconstructed views
    old_descriptors = []
    for old_view in done:
        old_descriptors.append(old_view['desc'])

    # match old descriptors against the descriptors in the new view
    # matcher.add(old_descriptors)
    matcher.train()
    # new_descriptor = np.array([view['desc']])
    matches = matcher.knnMatch(queryDescriptors=view['desc'], trainDesciptors=old_descriptors, k=2)

    p_2D = np.array([view['kp'][m.trainIdx].pt for m in matches]).reshape(-1, 1, 2).astype(np.float32)

    # compute new pose using solvePnPRansac
    _, R, t, _ = cv2.solvePnPRansac(points_3D[:, np.newaxis], p_2D[:, np.newaxis], K)  # 3d point 수정
    R, _ = cv2.Rodrigues(R)
    return R, t


def plot_points(done, points_3D):
    """Saves the reconstructed 3D points to ply files using Open3D"""
    results_path = './figure'
    os.makedirs(results_path, exist_ok=True)

    number = len(done)
    filename = os.path.join(results_path, str(number) + '_images.ply')
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3D)
    o3d.io.write_point_cloud(filename, pcd)


def reconstruct(K, view_l):
    points_3D = np.zeros((0, 3))
    done = []

    # baseline
    view1 = view_l[0]
    view2 = view_l[1]
    good_matches = match_keypoints(view1, view2)
    R, t, pts1, pts2 = essentialMat_estimation(view1['kp'], view2['kp'], good_matches, K)
    p3d = triangulate(R, t, K, pts1, pts2)
    done.append(view1)
    done.append(view2)
    points_3D = np.concatenate((points_3D, p3d.T), axis=0)
    p_3D_init = points_3D
    plot_points(done, points_3D)
    visualization(points_3D)

    # not baseline
    for i in range(2, len(view_l)):
        view2 = view_l[i]
        view2_R, view2_t = compute_PNP(K, view2, p_3D_init, done)
        print("R={}\nt={}".format(view2_R, view2_t))

        for i, old_view in enumerate(done):
            matches = match_keypoints(old_view, view2)
            kp1 = old_view['kp']
            kp2 = view2['kp']
            pts1 = np.array([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2).astype(np.float32)
            pts2 = np.array([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2).astype(np.float32)
            p3d = triangulate(view2_R, view2_t, K, pts1, pts2)
            points_3D = np.concatenate((points_3D, p3d.T), axis=0)

        done.append(view2)

    return points_3D


def run(path):
    K, img_files = read_files(img_root)
    view_l = create_view(img_files)
    points_3D = reconstruct(K, view_l)


if __name__ == '__main__':
    img_root = './data/'
    run(img_root)