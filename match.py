import os
from os import path as osp

import cv2
import numpy as np
import matplotlib.pyplot as plt


def extract_feature(img):
    feature = cv2.SIFT_create()
    kp, desc = feature.detectAndCompute(img, None)
    return kp, desc


def create_view(img_files):
    view_l = []
    for i in img_files:
        i_img = cv2.imread(i, cv2.COLOR_BGR2RGB)
        img_name = i.split('/')[-1][:-4]
        kp, desc = extract_feature(i_img)
        view = {'img': i_img, 'name': img_name, 'kp': kp, 'desc': desc}
        view_l.append(view)

    # draw keypoints
    img = view_l[0]['img']
    kp = view_l[0]['kp']
    img_draw = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(img_draw)
    # plt.show()
    return view_l


class Match:
    def __init__(self, view1, view2):
        self.view1 = view1
        self.view2 = view2
        self.img_name1 = view1['name']
        self.img_name2 = view2['name']
        self.indice1 = []  # indices of the matched keypoints in the first view
        self.indice2 = []
        self.distances = []  # distance between the matched keypoints in the first view
        self.inlier1 = []  # indices of the matched keypoints from the first view not removed using the essential matrix
        self.inlier2 = []
        self.matcher = cv2.BFMatcher_create()

    def get_matche(self, view1, view2):
        matches = self.matcher.knnMatch(view1['desc'], view2['desc'], k=2)  # len(matches): # of matching point
        good_matches = [m1 for m1, m2 in matches if m1.distance < 0.80 * m2.distance]  # 좋은 매칭 결과
        sort_matches = sorted(good_matches, key=lambda x: x.distance)

        for i in range(len(sort_matches)):
            self.indice1.append(sort_matches[i].queryIdx)
            self.indice2.append(sort_matches[i].trainIdx)
            self.distances.append(sort_matches[i].distance)
        return sort_matches

    def draw_matches(self, sort_matches):
        img_match = cv2.drawMatches(self.view1['img'], self.view1['kp'], self.view2['img'], self.view2['kp'],
                                    sort_matches, self.view2['img'], flags=2)
        plt.imshow(img_match)
        plt.show()


def create_all_matches(views):
    matches = {}
    for i in range(0, len(views) - 1):
        for j in range(i + 1, len(views)):
            match_ij = Match(views[i], views[j])
            match_ij.get_matche(views[i], views[j])
            matches[(views[i]['name'], views[j]['name'])] = match_ij
    return matches
