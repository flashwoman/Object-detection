# Preprocessing _book_



[TOC]

### 1. make Binary Image



### 2. cv.Sobel()

```python
# using Sobel()
tmp = cv.Sobel(img_gray, cv.CV_64F, 1, 0, ksize=5)
sobel64f = np.absolute(tmp)
sobel_united = np.uint8(sobel64f)
```



### 3. cv.canny()

```python
# using Canny()
cannied_img = cv.Canny(img_gray, 250, 500)
cannied_sobel = cv.Canny(tmp, 50, 500)
cannied_sobel_united = cv.Canny(sobel_united, 150, 200)
```

  

### 4. cv.morphologyEx()

**IMPORTANT**

- **kernel**
  - size
  - kernel shape

```python
# MorphologyEx (노이즈 제거하기)
# 범위 값으로 HSV 이미지에서 마스크를 생성합니다.
img_mask1 = cv.inRange(img_hsv, lower_color1, upper_color1)
img_mask2 = cv.inRange(img_hsv, lower_color2, upper_color2)
img_mask3 = cv.inRange(img_hsv, lower_color3, upper_color3)
img_mask = img_mask1 | img_mask2 | img_mask3
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))

img_mask = cv.morphologyEx(img_mask, cv.MORPH_OPEN, kernel)
img_mask = cv.morphologyEx(img_mask, cv.MORPH_CLOSE, kernel)
```



### 5. cv.threshold()

```python
# 5가지 threshold 방법
ret,thresh1 = cv.threshold(img_gray,127,255,cv.THRESH_BINARY)
ret,thresh2 = cv.threshold(img_gray,127,255,cv.THRESH_BINARY_INV) # 책 아웃라인 검출
ret,thresh3 = cv.threshold(img_gray,127,255,cv.THRESH_TRUNC)
ret,thresh4 = cv.threshold(img_gray,127,255,cv.THRESH_TOZERO) # 책장 프레임 검출
ret,thresh5 = cv.threshold(img_gray,127,255,cv.THRESH_TOZERO_INV)
```



### 6. cv.watershed()

