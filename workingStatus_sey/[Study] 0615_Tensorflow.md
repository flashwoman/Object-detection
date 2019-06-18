[TOC]



# TensorFlow 이용하기



### 목표

[Tensorflow 이용하지 않는다면...](<https://bradbury.tistory.com/64>)

1. 책장배경, 책장의 색조(range 설정 필요) 찾기
2. 찾은 색조 이미지에서 빼기 (이때 붙어있는 색조에서만 빼기... 책 중간에서 다량의색이 빠지는것을 방지)
3. 
   - [morphEx](https://docs.opencv.org/3.4/dd/dd7/tutorial_morph_lines_detection.html) :책에서 잘못 빠진 색조 반환 위해 dilate시키기
   - 책장을 뺀 이미지에서 Outline검출 후 원래이미지를 크롭시키기
4. 아라언니에게 이미지 넘기면 됨



### 데이터 학습 과정

[Reference 1](<https://github.com/andreasntr/ColorClassifier>)

[Reference 2](<https://webnautes.tistory.com/1256>)

1. 데이터셋 로드

   - 학습대상 데이터 변수저장
   - 라벨 one_hot 인코딩 시키기
   - 데이터사이즈 설정

2. train / test 데이터 분리

3. 인덱스 만들기

4. 데이터에 라벨링하기 : 학습데이터와 라벨에 인덱싱해주기

5. validation indexes .....????  : 유효성 검사인듯...(정답이 아닌것들을 뽑아서 넣어놓은것 같다_ 반대일수도...)

6. 학습, 테스트, 유효성 x_y 데이터를 np.savez_compressed 함수로 압축시키기

7. 머신러닝 모델 구축하기

   ```python
   model = tf.keras.models.Sequedntial([
       tf.keras.layers.Dense(64, input_shape(3,),activation=tf.nn.relu),
       tf.keras.layers.Dense(9, activation=tf.nn.softmax)
   ])
   ```

8. 머신러닝 모델 컴파일하기

   ```python
   model.compile(
   	optimizer=tf.train.AdamOptimizer(0.002),
   	loss='categorical_crossentropy',
   	metrics=['accuracy'])
   ```

9. 데이터 학습시키기

   ```python
   print("Training:")
   model.fit(colors_train, labels_train, epochs=10, batch_size=32)
   print("Training ended. Validating:")
   model.fit(colors_validation, labels_validation, epochs=10, batch_size=32)
   ```

10. 학습 모델 저장하기
    `model.save_weights("model_weights.h5")`
    h5는 대용량 데이터를 저장하기 위한 파일 포맷이다.



### 함수

- [.one_hot](<http://blog.naver.com/PostView.nhn?blogId=wideeyed&logNo=221164692553&beginTime=0&jumpingVid=&from=search&redirect=Log&widgetTypeCall=true&directAccess=false>) 
  - 인코딩
  - 인풋차원 + 1 = 아웃풋 차원
  - 아웃풋 차원에 대해 차원줄이기 필요 (reshape?)