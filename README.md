
# 강화학습을 이용한 인공지능 슈퍼마리오 만들기 

### 모두의연구소 flipped school 슈퍼마리오 대회편 


![image7](https://user-images.githubusercontent.com/11300712/37243439-a071562c-24bc-11e8-989b-4d55522b3bcf.jpg)


소개 

안녕하세요. 

저는 모두의 연구소 풀잎스쿨 (flipped school) 1기 강화학습 반에 참여한 정원석이라고 합니다. 


저희는 “파이썬과 케라스로 배우는 강화학습”이라는 책을 교재로 각자 사전학습을 한 뒤 매주 한번씩 모두연에서 만나,

궁금한한 내용을 질문하고 토론하며 즐겁게 공부하였습니다. 

이 교재를 통하여 Reinforcement learning의 기본개념과 알고리즘을 배우고 파이썬과 케라스를 이용하여 구현하였습니다.

이론과 실습을 같이 하면서,  강화학습의 이론이 문제에  어떻게 적용되는지 경험해보는 좋은 기회였습니다. 

공부를 하며,

“책에 나와있지 않은 다른 환경에서 우리가 배운 학습 알고리즘은 적용해보면 재밌겠다! ”
라는 욕심이 생겼서,

우리가 배운 강화학습 알고리즘을 슈퍼마리오 환경에 적용해보기로 하였습니다. 


### 그래서 생겨난 모두의연구소 풀잎스쿨 "슈퍼마리오대회" 


![image14](https://user-images.githubusercontent.com/11300712/37243443-c592f60e-24bc-11e8-9c93-4f736c5ee45a.jpg) 


위의 스케줄과 같이 풀잎스쿨 3개월 코스 마지막날에 다같이 모여, 각자만든 슈퍼마리오로 대결을 합니다. 



강화학습으로 똑똑한(?) 슈퍼마리오를 만들며 여러가지 이슈들이 많이 발생하였고 슈퍼마리오 환경을 이해하기 위해 1기분들이 많은 고생을 하였습니다. 

그래서,

다음 기수 분들 혹은 슈퍼마리오 환경을 강화학습으로 풀어보기 위한 분들을 위하여

슈퍼마리오 메뉴얼을 제작하였습니다. 

개선해야할 부분이 있다면
email :  wonseokjung@hotmail.com 
facebook:wonseokjung@hotmail.com

으로 연락주시면 참고하겠습니다. 


_ _ _

목차

1. 강화학습 기초 이론 및 슈퍼마리오 구성 요소
1.1 . Envrionment 
1.2 Emulator 
1.3. Algorithm 
1.4. 프로그래밍 언어 

2. 설치 매뉴얼    
2.1 python 설치
2.2 Emulator 설치
2.3 Environment설치 

3. Algorithm 설명 

4. 오류 해결

5. 슈퍼마리오 훈련
5.1 openAI 함수 설명
5.2 슈퍼마리오의 action

6. 대회규칙



_ _ _

## 1. 강화학습 기초 이론 및 슈퍼마리오 구성 요소

똑똑한(?) 슈퍼마리오를 만들기 위해서는 4가지가 필요합니다.

### 1.1 . Envrionment 
![image1](https://user-images.githubusercontent.com/11300712/37243505-b82841f8-24bd-11e8-99a0-a69fe6faa822.jpg)


Agent인 마리오가 Environment와 상호작용을 하며 환경에서 주어진 action중 하나를 선택하고 그 action을 선택함으로써 reward(보상) 을 받게 됩니다.

이러한 Frame을 MDP라고 합니다.

MDP는 목표를 달성하기 위해 상호작용하는 큰 frame입니다.

여기서 배우고 결정을 내리는 것을 Agent 라고 합니다.

Agent와 상호작용하는 것, agent를 제외한 모든 것을 Environment라고 합니다.

Agent는 각 time step $$t$$마다 환경을 표현하는 state $$s$$를 받습니다. 
 
State를 받고 Environment가 상호작용을 하면서 agent가 action을 선택하면,

Environment는 agent의 action에 응답해 그에 맞는 새로운 상황과 reward를 agent 에게 줍니다.

강화학습을 이용하여 스스로 높은 reward를 받는 action을 선택하는 슈퍼마리오를 만드는 것이 목표입니다. 

이러한 정보가 있는 “슈퍼마리오”의 Environment가 필요합니다. 


### 1.2 Emulator 

두번째로 슈퍼마리오를 실행하기 위한 Emulator가 필요합니다. 

강화학습에서는 Agent가 현재의 state에서 action을 선택하고, 

그 action을 하여 받는, 다음의 state와 reward 정보를 받으며 학습을 합니다. 


여기서 Agent는 마리오이고 state는 게임 화면입니다. 

마리오는 이미지 데이터인 게임 화면을 matrix로 바꿔 state를 인식하고 이것을 이용하여 학습을 합니다. 

그렇게 하기 위해서는 슈퍼마리오 게임을 구동하는 프로그램이 필요한데요. 

이 메뉴얼에서 우리는 fceux라는 emulator을 사용할 것입니다.


### 1.3. Algorithm 

강화학습에서 목표는, Agent가 학습을 하며 받는 총 reward의 크기를 최대화하는 것입니다. 
바로 앞에서 받을수 있는 reward를 최대화 시키는 것이 아닌 long run에서의 축척된 reward를 최대화시키는 것 입니다.
그럼 reward를 최대화 시키기 위해 그 환경에 맞는 적절한 학습 알고리즘을  사용하여야 합니다. 




### 1.4. 프로그래밍 언어 

환경을 구성하고 강화학습의 알고리즘을 구현하기 위해 Python 프로그래밍 언어를 사용합니다. 

그리고 

이 메뉴얼은 Window 환경에서는 슈퍼마리오 환경, emulator와 충돌이 많아 Ubuntu와 Mac을 기준으로 작성하였습니다. 

UBUNTU, MAC에서 실습하실것을 강력 추천 드립니다( 윈도우에서 성공한 사례는 보지 못하였습니다.)


## 2. 설치 매뉴얼

### 2.1 python 설치

![image11](https://user-images.githubusercontent.com/11300712/37243743-ea42d1ae-24c1-11e8-9310-2ccad3af6ecd.jpg)

a.첫번째로 파이썬을 설치하셔야 합니다. 

https://www.python.org/ - python 홈페이지 

위의 홈페이지에 들어가셔서 Mac 혹은 우분투 버전을 다운로드 후 설치해주세요. 

파이썬은 버전이 여러가지 있는데, 

현재 파이썬 3.5버전에서의 실행을 확인하였습니다. 

그러므로 파이썬 3.5버전 설치를 권장합니다. 


b.그리고, 딥러닝 모델을 쉽게 구현할 수 있는 라이브러리인 케라스와 텐서플로우를 설치해주세요. 
https://www.tensorflow.org/install/ - Tensorflow 

Ubuntu ctrl+alt+t 를 눌러 커맨창에서 

tensorflow 일반 버전은

`sudo pip3 install tensorflow `

혹시 gpu가 있으시다면, 

`sudo pip3 install tensorflow-gpu`  명령어로 tensorflow를 설치해주세요. 

케라스 홈페이지를 다음과 같으며 케라스 설치하는 방법이 자세히 설명되어 있습니다. 

https://keras.io/#installation

c. Jupyter notebook을 설치해주세요. 

실습편에서 jupyter notebook을 이용하여 코드 설명을 하였습니다. 

이 파일을 실행시키기 위하여 jupyter notebook이 필요합니다. 

http://jupyter.org/install

`python3 -m pip install --upgrade pip`
`python3 -m pip install jupyter`

위의 명령어를 이용하여 설치하여 주세요. 


## 2.2 Emulator 설치

![image5](https://user-images.githubusercontent.com/11300712/37247381-3728c70e-24fd-11e8-9d5e-336b0d697e1f.jpg)



Emulator인 fceux는 슈퍼마리오 게임을 실행시켜주는 역할을 합니다. 

fceux홈페이지에 들어가시면 emulator에 관한 자세한 설명이 나와있습니다. 
http://www.fceux.com/web/home.html 

ubuntu 혹은 Mac에서 커맨드 창에 다음의 명령어를 입력하여 fceux를 설치하여주세요. 

Ubuntu 

`sudo apt-get update`
`sudo apt-get install fceux`

MAC 

https://brew.sh/  - homebrew 웹사이트

커맨드 창에 아래의 명령어 입력
`brew install fceux`
`sudo apt-get install fceux`

## 2.3 Environment설치 

![image3](https://user-images.githubusercontent.com/11300712/37247379-3069fc1c-24fd-11e8-9f2c-4196db6bf4a5.jpg)

Openai 에서 gym 과 baselines를 제공해 줍니다. 

우리는 이를 이용하여 강화학습의 환경을 다운 받고 학습 알고리즘을 실험해 볼 수 있습니다. 

openai에서 슈퍼마리오의 환경은 제공하고 있지 않지만, 

우리는 슈퍼마리오의 환경을 gym의 폴더에 넣어 openai가 제공하는 함수를 사용하기 위해 먼저 openAI의 gym 라이브러리를 다운받아야 합니다. 



https://openai.com/   - Openai 

https://github.com/openai/gym - gym git 주소





a.필요한 package를 설치하세요.

MAC

`brew install cmake boost boost-python sdl2 swig wget`

Ubuntu

`apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig`


b.gym 설치


ubuntu

pip를 이용해서 설치하기

`pip3 install gym` 

혹은 git에서 다운받아 설치하기 

`git clone https://github.com/openai/gym.git`
`cd gym`
`pip3  install -e`


baselines에 DQN,A3C등 여러가지 알고리즘의 예제가 나와있습니다. 

우리는 baselines를 직접 사용하진 않지만 강화학습 알고리즘 분석을 위해 다운받길 권장합니다. 


Baselines

https://github.com/openai/baselines

pip를 이용해서 설치하기

pip3 install baselines


혹은 git에서 다운받아 설치하기 

git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .


![image5](https://user-images.githubusercontent.com/11300712/37247381-3728c70e-24fd-11e8-9d5e-336b0d697e1f.jpg)


다음은 슈퍼마리오의 환경을 다운받아야 합니다. 

강화학습으로 슈퍼마리오를 학습시킬 수 있도록, 

Philip Paquette가 슈퍼마리오와 둠의 환경을 github에 업로드 해놓았습니다. 

우리는 gym-pull이란 함수를 이용하여 philip paquette가 만든 슈퍼마리오 환경을 위에서 설치한 gym의 환경에 넣어줍니다. 


philip paquette의 github 주소

https://github.com/ppaquette/gym-super-mario

커맨드창을 여신후에 

a.pip를 이용하여 gym-pull을 설치하세요 

`pip3 install gym-pull`

b. 커맨드창에서 python3을 이용하여 python3에 들어가신 후 gym과 gym_pull을 import 하셔야합니다.  
- sudo python3 명령어를 이용하여 python3에 들어가세요 
- python3에서 gym과 gym_pull 라이브러리를 import 합니다. 
`import gym`
`import gym_pull`

c. gym_pull을 이용하여 ppaqutte의 supermario 환경을 가져옵니다. 

`gym_pull.pull('github.com/ppaquette/gym-super-mario')`       

d.gym.make의 함수를 이용하여 슈퍼마리오 환경이 로드 되는지 확인합니다. 

`env = gym.make('ppaquette/SuperMarioBros-1-1-v0')`

* gym_pull error가 발생하시는 분은 메뉴얼 목차 4의 오류발생 부분을 확인해주세요.
 
## 3.Algorithm 설명 

![image9](https://user-images.githubusercontent.com/11300712/37243738-db1a797a-24c1-11e8-8be6-d68387531fca.jpg)

저는 학습알고리즘으로 DQN을 사용하였습니다. 
DQN은 replay memory를 이용하여 state,action,reward, next state의 정보를 memory에 저장하고, 이 정보를 이용하여 convolutional neural network라는 딥러닝 모델을 사용하여 좋은 action을 선택하게하는 알고리즘 입니다. 

더 자세한 설명은 아래의 링크를 참조하세요. 

[DQN이란 무엇일까요? ](https://wonseokjung.github.io//rl_paper/update/RL-PP-DQN/)


참고로 저의 경우, 레벨1을 클리어하기 위하여 5000 에피소드 정도의 학습이 필요하였습니다.

DQN을 사용한 것은 단 하나의 예 입니다. 

환경에 맞는 적절한 강화학습 알고리즘을 사용하는 것이 중요하므로, 학습 알고리즘은 본인의 결정에 따라 자유롭게 선택하시면 됩니다. 


## 4. 오류 해결


A. 
설치하며 여러가지 오류가 발생하였는데요. 그중 가장 많이 발생한 gym_pull 오류를 풀잎1기 강화학습반 김경환님이 해결하신 방법입니다.

![image10](https://user-images.githubusercontent.com/11300712/37247567-60d718d2-2500-11e8-878c-451805bff4f9.jpg)
![image13](https://user-images.githubusercontent.com/11300712/37247569-6c597786-2500-11e8-801f-c690fa884118.jpg)
![screenshot from 2018-03-11 08-55-45](https://user-images.githubusercontent.com/11300712/37248031-036d15fc-250a-11e8-9827-df778a9d74e7.png)
![image12](https://user-images.githubusercontent.com/11300712/37248033-0bc40b34-250a-11e8-9fe4-7d04d0f3ae22.jpg)

![screenshot from 2018-03-11 08-56-36](https://user-images.githubusercontent.com/11300712/37248040-4afe19ac-250a-11e8-93d0-1a7060586067.png)


B. 풀잎2기 강화학습반 권력환님의 gym_pull 을 사용하지 않고 환경 가져오는 방법


gym & super mario 환경 설치

gym-0.9.4 설치
(gym이 이미 깔려있는 경우..)
`pip3 uninstall gym` 
pip3 install gym==0.9.4  

## baseline 설치
$ pip3 install baselines

## Super Mario Gym 환경 설치 
$ git clone https://github.com/ppaquette/gym-super-mario.git
$ cd gym-super-mario/
$ pip3 install -e .

## Super Mario Gym 환경 불러오기
$ python3 -V
Python 3.5.2

$ python3
>> import gym
>> import ppaquette_gym_super_mario
>> env = gym.make('ppaquette/SuperMarioBros-1-1-v0')
>> env.reset() 
>> env.close()

