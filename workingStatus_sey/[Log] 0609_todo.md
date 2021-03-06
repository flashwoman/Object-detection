### ~0609

- **해야할 것**

  - [책 Outline 검출](<http://hk.voidcc.com/question/p-ezrgawjn-gx.html>)

  - cv2.findHomography

  - cv2.warpPerspective

  - [책 한권씩 검출](<https://sosal.kr/1067>)

    - 각각을 좌표리스트로 split
    - 각각의 색 검출

  - 설정한 색채 알고리즘을 기반으로 책 재배열

    - SOM
    - K-Means

    

    <br>

    

- 진행상황

  - test_shapeDetector.py

    1. 올바른 도형검출 불가

       - 장애요소 : 음영, 책장과 책의 색채대비 미비, 책의경계모호

       - 개선방향 : 색으로 책장먼저 detect후 ignore시키기 -> outline 따기

- 시도해 볼 수 있는 것

  - [findHomography](<https://m.blog.naver.com/PostView.nhn?blogId=dlcksgod1&logNo=221295478427&proxyReferer=https%3A%2F%2Fwww.google.com%2F>) : 특징매치, 복잡한이미지에서 객체검출
    - 정확한 검출을 위해 cv2.perspectiveTransform() 사용 (적어도 정확한 4개의점 필요)
    - 가장 좋은 연결점은 ratio test이용
  - [morphologyEx](<https://m.blog.naver.com/samsjang/220505815055>) : 이미지형태변환(이미지 정리_굵게, 가늘게_하기 정도로 이해)
  - [Sobel()]() : 가로선과 세로선을 따로 찾기위해서
  - [Canny()](<https://m.blog.naver.com/samsjang/220507996391>) : 노이즈제거 + Gradient 탐색 + 최대-최소값 + Thresholding
  - [Feature Matching](<https://m.blog.naver.com/dlcksgod1/221295457905>)
  - [getPerspectiveTransform()](<https://opencv-python.readthedocs.io/en/latest/doc/10.imageTransformation/imageTransformation.html>) : 이미지의 기하학적 변경
  - [ColorClassifier](<https://github.com/andreasntr/ColorClassifier>) using Tensorflow : 텐서플로우 사용해서 선택한 픽셀의 색 인식하기(80%정확도 가지는 코드)

- OpenCV 기본이해

  - [cv2.imread()](<https://opencv-python.readthedocs.io/en/latest/doc/01.imageStart/imageStart.html>)