[TOC]

### Code Sequence



<br>



1. Class **Node**정의
   	: Feature Vetor , Prediction Vector를 랜덤하게 정의해준다.

2. Class **SOM**정의
   - SOM 모델 정의 : height, width, feature-vector size, prediction-vector size, radius, learning_rate
     - **self.height**
     - **self.width**
     - **self.radius** 
     - **self.total** 
     - **self.learning_rate**
     - **self.nodes**
     - **self.FV_size**
     - **self.PV_size**
   - SOM.train() : train 최대반복횟수, train_vector
     - time_constant
     - radius_decaying
     - learning_rate_decaying
     - influence
     - stack
   - SOM.predict() : Feature Vector
     - best
   - SOM.best_match() : target feature vector
     - minimum
     - minimum_index
   - SOM.FV_distance() : feature-vector 1, feature-vector 2
     - distance    *between FV_1, FV_2*
   - SOM.distance() : node 1, node 2
     - distance    *between node1, node2*
3. 