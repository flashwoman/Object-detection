### WaterShed

: 책장에서 책더미 분리 시키기

 



*작업이미지크기 : 459 * 612 기준 * 

1. Gray& Blur작업 (노이즈제거)

2. Threshold ( THRESH_BINARY + THRESH_OTSU ) : 윤곽선은 강조해주는 로직

3. kernel (3,3) -> kernel ( 13, 3 ) : 세로방향의 책들 검출 

   * (21,3)은 명확하지만 덜 검출

   

![](C:\dev\Object-detection\MD\image\image1.PNG)

4. cv.distanceTransform : 배경으로부터 멀어질 수록 짙은 흰색으로 표시