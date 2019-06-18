# K-Means 알고리즘

*for **clustering***



**Reference Pages**

- [K-means *vs* SOM](<http://jaynewho.com/post/7>)



[TOC]

### 학습과정

1. Cluster 갯수 **K** 설정하기
2. Select at random K points, the centroids (not necessarily from your dataset)
3. 각 데이터를 가장 가까운 클러스터의 중심점으로 할당한다.
   Assign each data point to the closest cenroid --> That forms K clusters
4. K개로 나뉜 각 군집의 새로운 중심점을 찾아 새로운 Cluster의 중심점으로 지정합니다.
5. Cluster 중심점들이 더이상 이동하지 않을 때까지 Step 3-4를 반복합니다.