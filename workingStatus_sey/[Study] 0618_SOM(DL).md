# SOM

[TOC]

**SOM : Self - Organizing Map**(자기조직화 지도)

0. Reference Page Lists
   - [0. ANN](<https://untitledtblog.tistory.com/141>)
   - [1. 간략소개](<https://neosla.tistory.com/25>)
   - [2. SOM_ratsgo's blog](<https://ratsgo.github.io/machine%20learning/2017/05/01/SOM/>)
   - [3. 코드구현과정 reference](<https://untitledtblog.tistory.com/5>)
   - [4. 기타 코드 레퍼런스 -1](<http://jaynewho.com/post/7>)
   - 



### SOM 소개

- **ANN**(Artificial Neural Network) 신경망 알고리즘
- 비지도학습
- 사용 : **Clustering** + Classification
- SOM의 특징
  - Input 데이터들 사이의 위상을 잘 나타낸다
  - 잘 구별되지 않는 데이터간의 상관관계를 찾아낼 수 있다
  - 비지도학습으로 Clustering을 수행할 수 있다
  - 비지도이고, Label Data가 없으므로, 역전파(Back-propagation)과정도 없다
  - Output node간에 후속 연결이 필요하지 않다



<br>



### 알고리즘 전개

- 학습방법 : 
  비지도학습방법을 기반으로, 각 뉴런끼리의 **가중치**들이 서로 **경쟁학습**(competitive)을 하게 된다.

  * 밑의 그림은 SOM의 뉴런들의 가중치가 업데이트되며 학습하는 과정을 나타내었다.
    Input data 는 random으로 선택되며,
    선택된 데이터와 각 뉴런(경쟁층의 노드)들의 거리를 계산한다.
    **거리(위치)**를 기반으로 뉴런의가중치가 업데이트 된다.
    입력된 모든 데이터의 가중치가 업데이트 될 때까지 수행된다.

  * 가중치 업데이트 Logic : 가까운 뉴런은 더 가깝게, 먼 뉴런은 더 멀게 가중치가 주어진다.(**Instance-based Learning**)

    ![img](https://t1.daumcdn.net/cfile/tistory/991006455C286ED31F)

    ![img](http://i.imgur.com/eHUVAtr.gif)



<br>



### 결과값 : 고차원의 저차원화

- n차원 -> 저차원 :
  임의의 n차원 입력벡터가 들어왔을 때, 가장 가까운 격자벡터를 찾습니다.(**Winning Node**) --> 이 벡터에 대응되는 2차원상 격자에 해당 입력벡터를 할당하면 이것이 바로 군집화가 되는 것입니다.

  - 결과물 이미지

    고차원공간의 원데이터가 25개 격자에 할당(군집화)되어 있고, 동일한 격자에 할당된 입력값끼리도 그 위치가 서로 다르게 임베딩 되어있음을 확인할 수 있다.

    ![img](http://i.imgur.com/EE8NF6J.png)

- Input Data 차원에 따른 결과값 도식화

  - 결과물 이미지 -2

    

    ![img](https://t1.daumcdn.net/cfile/tistory/990BCD505C287A7838)



<br>



### 코드 구현과정

[Reference Page](<https://untitledtblog.tistory.com/5>)



0. 목적 : **Clustering**

1. 구성

   - 입력층 (input layer) : 입력 벡터 입력받는 층

   - 경쟁층 (competitive layer) : 입력벡터의 특성에 따라 입력 벡터가 한 점으로 클러스터링 되는 층 
     __그럼 이게 중간에 위치되어있는건가....?_

   - 가중치 (weight) : 인공신경망에서 가중치는 각 입력 값에 대한 입력 값의 중요도 값을 말한다.

   - 노드 (node) : 경쟁층에서 입력 벡터들이 서로의 유사성에 의해 모이는 하나의 **영역**


     ![img](https://t1.daumcdn.net/cfile/tistory/25321F4C568D1BA033)

2. 알고리즘 구조

   SOM에서 알고리즘을 실행하거나 종료하기 위해서는...
   **현재 입력 벡터, 현재 반복 횟수** 두 개의 값을 유지해야 한다.

   1. 가중치 행렬 각 원소의 값을 0보다 크고 1보다 작은 임의의 값으로 초기화

   2. 입력 벡터($x_i$)와 경쟁층에 존재하는 j개의 노드간의 **거리**($D_{ij}​$)를 계산

   3. 현재 입력 벡터에서 $D{ij}$값이 가장 작은 경쟁층의 노드를 선택

   4. 3을 바탕으로 해당 노드의 가중치와 이웃 노드의 가중치를 수정

   5. 현재 입력 벡터가 마지막 입력 벡터라면 다음 과정으로 이동하고, 그렇지 않다면 과정 2~4를 반복한다.

   6. *현재 반복 횟수가 최대 반복 횟수라면 알고리즘을 종료한다.*

   7. *현재 반복 횟수가 최대 반복횟수가 아니라면 현재 입력 벡터를 처음 입력 벡터로 설정하고 과정 2~4를 반복한다.*

      <br>

      ![img](https://t1.daumcdn.net/cfile/tistory/2765114C568E790A04)

      

      <br>

3. **$D{ij}​$ (입력벡터-경쟁층노드 간의 거리)**와 **노드의 가중치**를 수정하는 연산 정의

   - $D{ij}$ (입력벡터-경쟁층노드 간의 거리) 수정
     

     $D_{ij} = \sum_{i=1}^{n}(w_{ij} - x_i)^2$  


     위의 수식에서 $n$은 입력 벡터의 크기, $w_{ij}$는 가중치 테이블에서 $i$행 $j$열의 값을 나타낸다. 그리고 가중치와 뺄셈 연산을 하는 $x_i$ 는 입력 벡터의 $i$번째 값을 뜻한다.

   - 해당 노드의 **가중치** 수정


     $w_{ij}(new) = w_{ij}(old) + \alpha(t)(x_i - w_{ij}(old))​$

      $w_{ij}(new)$는 **새로운 가중치**, $w_{ij}(old)$는 이전 가중치를 뜻한다. 그리고 우변의 두번째 항에 있는 $\alpha(t)$는 **학습률**을 나타내며, 알고리즘을 설계할 때 정하는 값으로 알고리즘이 반복될 수록 값이 **작아진다**.



<br>



### 코드 Reference

1. 4개의 입력 벡터를 2개의 그룹으로 클러스터링하는 SOM 설계하기.
   [Reference Page](<https://untitledtblog.tistory.com/5>)

   - 입력벡터 (input data)
     

     $$\left[\begin{array}{rrr}1&0&0&1\\0&0&1&1\\0&1&0&1\\0&0&1&0\end{array}\right]​$$
     

   - 경쟁층 (competitive layer) : 노드의 수 **2** 
     (입력 벡터를 2개의 그룹으로 클러스터링 하기 때문)

   - 최대 반복 횟수 설정하기 : **10,000**번

   - 초기 학습률 설정 $\alpha(t)​$ : **0.6**

   - 학습 반복시 학습률 수정 연산 설정 $\alpha (t+1)$

     $\alpha (t+1) = 0.4  *  \alpha(t)​$

   - 이웃노드 : 경쟁층의 노드 수가 2개 뿐이기 때문에 이웃 노드의 개념은 없다고 가정한다. 

     _출력층 노드의 수가 많은 자기조직화지도에서는 가중치가 수정되는 노드의 이웃 노드도 같이 가중치를 수정해준다._

   - 초기학습률과 학습률의 감소 비율은 알고리즘을 설계할 때, 직관적으로 설정하는 값이다.
     이를 이론적으로 설정하고 싶다면, **퍼지이론(Fuzzy theory)**를 참조하면 된다.



<br>



2. 알고리즘 설계

   - 초기 가중치 행렬 초기화
     : 0~1 범위에 있는 임의의 값으로 초기화한다.

     - 가중치 행렬 크기 : 4 X 2 (입력벡터의 크기 _4_ X 출력층 노드의 수 _2_)
       

       $$\left[\begin{array}{rrr}0.5&0.1\\0.3&0.7\\0.6&0.8\\0.2&0.2\end{array}\right]​$$
       

   - 입력 벡터(1, 0, 0, 1)와 경쟁층 노드의 거리 계산 및 가중치 수정

     ![img](https://t1.daumcdn.net/cfile/tistory/2344654D568E175303)![img](https://t1.daumcdn.net/cfile/tistory/27210F34568E17B305)

     > 첫번째 입력 벡터는 경쟁층의 첫번째 노드와 가장 가깝다.
     > 첫번째 노드가 선택되었으므로,  첫번째 노드에 대한 가중치를 수정한다.

   - 입력 벡터에 대한 노드의 가중치 변경하기 (예: $w_{12}​$_첫번째노드의 두번째 가중치_)

     ![img](https://t1.daumcdn.net/cfile/tistory/2416E13D568D1DC82B)
     

     ![img](https://t1.daumcdn.net/cfile/tistory/2773E435568E18FE0D)
     

     > $w_{12}$는 기존 0.3에서 **0.12**로 변경되었다.


     첫번째 노드의 가중치를 모두 변경하면 밑의 행렬식이 나온다.

     $$\left[\begin{array}{rrr}0.8&0.1\\0.12&0.7\\0.24&0.8\\0.68&0.2\end{array}\right]$$
     

     두번째 입력벡터에 대한 노드(2번째 노드)의 가중치를 모두 변경하면 밑의 행렬식이 나온다.

     $$\left[\begin{array}{rrr}0.8&0.04\\0.12&0.28\\0.24&0.92\\0.68&0.68\end{array}\right]$$
     


     세번째 입력벡터에 대한 노드(2번째 노드)의 가중치를 모두 변경하면 밑의 행렬식이 나온다.

     $$\left[\begin{array}{rrr}0.8&0.016\\0.12&0.712\\0.24&0.368\\0.68&0.872\end{array}\right]$$
     


     네번째 입력벡터에 대한 노드(2번째 노드)의 가중치를 모두 변경하면 밑의 행렬식이 나온다.

     $$\left[\begin{array}{rrr}0.8&0.0064\\0.12&0.2848\\0.24&0.7472\\0.68&0.3488\end{array}\right]​$$

     

   - 학습률 수정

     ![img](C:\Users\DELL\PycharmProjects\Object-detection\workingStatus_sey\233E6A3D568D1DB406)

     > 학습률은 0.6에서 **0.24**로 수정된다.

     

   <br>

   

3. 



<br>



### 아직 파악하지 못한 것들

- [x] ~~역전파 알고리즘의 필요성(Backpropagation Algorithm) --> 필요없다~~

- [ ] 어떻게 SOM을 double-layer화 시키지...?











