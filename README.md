# 3D Reconstruction using Structure from Motion
## process
1. Extract keypoints and feature descriptors from images
2. Match features between images
3. Recover pose of the baseline
4. Reconstruct 3D points (Triangulate)
5. Recover pose of the next view using Perspective-n-Point
6. Reconstruct the next set of points
7. Plot points

## Results
* ### Keypoints
![keypoints](https://user-images.githubusercontent.com/97673250/235083711-3a7fc144-6399-442f-90f8-e04679a9953d.png)
* ### Matching
![matches](https://user-images.githubusercontent.com/97673250/235083966-68924a24-3ec9-4f8a-b1f8-c1f7bbbf3f31.png)
* ### Triangulate
1. Reconstruction after 2 images
![3dpoints](https://user-images.githubusercontent.com/97673250/235086468-59ac5610-f333-44fe-b9ea-1dc4297258b8.png)
