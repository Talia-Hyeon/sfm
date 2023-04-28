import os

import numpy as np
import cv2
import open3d as o3d


def remove_outlier_using_E(K, view1, view2, match):
    kp1 = view1['kp']
    kp2 = view2['kp']
    pixel_p1 = np.array([kp.pt for kp in kp1])[match.indice1]
    pixel_p2 = np.array([kp.pt for kp in kp2])[match.indice2]
    print("pixel_p2's shape: {}".format(pixel_p1.shape))
    # matches의 class(dematch)의 attribution: queryIndex, trainIndex, ...
    # queryIndex: 1번 img keypoint 번호
    # trainIndex: 2번 img keypoint 번호
    # reshape(len(matche),1,2)
    # pts'shape:(# of matches, 1, 2)  2->x&y 좌표

    E, mask = cv2.findEssentialMat(pixel_p1, pixel_p2, cameraMatrix=K, method=cv2.RANSAC)

    # Removes points that have already been reconstructed in the completed views
    match.inlier1 = pixel_p1[mask.ravel() == 1]  # img 1 inlier
    match.inlier2 = pixel_p2[mask.ravel() == 1]  # img 2 inlier
    return E


def triangulate(R, t, K, p1, p2):
    Rt0 = np.hstack((np.eye(3), np.zeros((3, 1))))  # 1st camera coordinate: world coordinate
    Rt0 = K @ Rt0
    Rt1 = np.hstack((R, t))
    Rt1 = np.matmul(K, Rt1)

    pt1 = np.transpose(p1)
    pt2 = np.transpose(p2)
    pt1 = np.squeeze(pt1)
    pt2 = np.squeeze(pt2)

    p3d = cv2.triangulatePoints(Rt0, Rt1, pt1, pt2)
    p3d /= p3d[3]  # Homogeneous Coordinate: [[[x]] [[y]] [[z]] [[1]]]
    # p3d's shape: (4,1,80)
    p3d = np.squeeze(p3d)[:3]  # (4,80)
    return p3d


def compute_PNP(K, view1, view2, exist_3D, done):
    matches = match_keypoints(view1, view2)

    # build corresponding array of 2D points and 3D points
    points_3D, points_2D = np.zeros((0, 3)), np.zeros((0, 2))
    for match in matches:
        old_image_idx, new_image_kp_idx, old_image_kp_idx = match.imgIdx, match.queryIdx, match.trainIdx
        if (old_image_idx, old_image_kp_idx) in point_map:
            # obtain the 2D point from match
            point_2D = np.array(view2['kp'][new_image_kp_idx].pt).T.reshape((1, 2))
            points_2D = np.concatenate((points_2D, point_2D), axis=0)

            # obtain the 3D point from the point_map
            point_3D = exist_3D[point_map[(old_image_idx, old_image_kp_idx)], :].T.reshape((1, 3))
            points_3D = np.concatenate((points_3D, point_3D), axis=0)

    # compute new pose using solvePnPRansac
    _, R, t, _ = cv2.solvePnPRansac(points_3D[:, np.newaxis], points_2D[:, np.newaxis], K)  # 3d point 수정
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


def reconstruct(K, view_l, all_matches):
    points_3D = np.zeros((0, 3))
    done = []
    point_map = {}

    # baseline
    view1 = view_l[0]
    view2 = view_l[1]
    match_object = all_matches[(view1['name'], view2['name'])]
    E = remove_outlier_using_E(K, view1, view2, match_object)
    retval, R, t, mask = cv2.recoverPose(E, match_object.inlier1, match_object.inlier2, K)
    p3d = triangulate(R, t, K, match_object.inlier1, match_object.inlier2)
    done.append(view1)
    done.append(view2)
    points_3D = np.concatenate((points_3D, p3d.T), axis=0)
    plot_points(done, points_3D)

    # not baseline
    for i in range(2, len(view_l)):
        view2 = view_l[i]
        view2_R, view2_t = compute_PNP(K, view2, points_3D, done)
        print("R={}\nt={}".format(view2_R, view2_t))

        for i, old_view in enumerate(done):
            matches = match_keypoints(old_view, view2)
            kp1 = old_view['kp']
            kp2 = view2['kp']
            pts1 = np.array([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2).astype(np.float32)
            pts2 = np.array([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2).astype(np.float32)
            p3d = triangulate(view2_R, view2_t, K, pts1, pts2)
            points_3D = np.concatenate((points_3D, p3d.T), axis=0)

        done.append(view2)

    return points_3D
