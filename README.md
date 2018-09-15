## Introduction
Playing-Atari-with-Deep-Reinforcement-Learning---Deep-Mind-Technologies 논문의 간단한 리뷰 입니다..
Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments 논문의 이해를 위해 기반이 되는 내용을 담고 있습니다.

## Abstract
강화학습을 사용하여 high-dimensional sensory input으로부터 policy(정책)들을 조절한은 법을 학습하기 위한 모델을 제안합니다.  
사용되는 모델은 Convolutional Neural Network (CNN) 이며, Q-learning의 변형모델이 사용됩니다. Input은 raw pixels 가되고 output은 future rewards의 추정값인 value function이 됩니다.  
Atari 2600 게임에 제안한 알고리즘을 적용시켜서 뛰어난 성능을 보이는 것을 알 수 있었습니다.

## Introduction
> 정확한 번역이 아닌 의역이 다수 존재합니다. 
 
vision과 speech 분야와 같이 high-dimensional sensory input으로부터 agent를 조작하는 법을 학습하는 것은 RL의 오랜 도전과제 중 하나였습니다. 기존의 연구들에서는 이 문제를 해결하기 위해서 hand-crafted features를 사용하였습니다. 그 결과, 이런 시스템이 보여주는 성능은 feature의 quality에 크게 영향을 받았습니다. 

최근 deep learning의 발전은 raw sensory data로부터 high-level feature들을 추출하는 것이 가능해 졌습니다. 그 결과, cnn, multi-layer perceptrons, restricted Boltzmann machine, rnn 등의 neural network architecture들의 이용을 통해서 RL에 새로운 방향성을 제시해 줄 것이라고 생각 하였습니다.

그러나, 강화학습은 deep learning의 관점에서 몇가지 어려움이 있습니다.  
1. RL은 sparse, noisy, delayed되어 발생하는 scalar reward로부터 학습이 진행됩니다. action과 reward사이의 delay는 deep learning 관점에서 input과 target(label) 의 관계성을 모호하게 만듭니다.  
> input 에 대해서 label 값이 바로 나오지 않아 몇차례의 time step이 필요하게 되어 label까지 여러 시도가 필요해진다.  
2. RL은 highly-correlated state들의 연속된 모음 ( state의 sequences ) 들을 만나게 된다. 이는 deep learning이 모든 데이터가 독립적이라는 가정과 상반된다.  
3. RL에서 데이터의 분포는 RL이 새로운 행동을 학습할때마다 변화하게 된다. 이는 고정된 분포에서의 데이터 발생을 가정하는 deep learning에게 문제가 될 수 있는 요소이다.
> 예를들어 새로운 정책을 학습하면 새로운 데이터가 생겨나고 그에따라 분포가 변할 수 있다. 

이 논문에서는 convolutional neural network를 통해서 위의 3가지 어려움을 극복하고 raw video data에서 agent의 control policy (정책)을 성공적으로 학습할 수 있다는 것을 보여줍니다. 모델은 Q-learning의 변형 모델이 사용되며, stochastic gradient descent 방법이 weight를 학습하기 위해서 사용됩니다. 

non-stationary distribution (3번 문제) 과 correlated data (2번 문제) 의 해결을 위해서 "Experience replay mechanism"방법을 사용합니다. 이 방법은 이전의 행동 샘플들로 부터 random하게 샘플을 가져오는 것으로 training data distribution을 완화 (smooth) 시켜 줍니다.

## Background
1. environment ![](https://user-images.githubusercontent.com/40893452/44390499-5ff6ed00-a568-11e8-9259-0c24668359a4.png) 와 상호작용하는 agent 의 task가 존재합니다.
2. 매 time-step마다 agent는 action ![](https://user-images.githubusercontent.com/40893452/44390581-9af92080-a568-11e8-892d-77664e9d06a8.png)을 선택합니다.
3. action은 game 내에서 선택 가능한 set of legal game actions ![](https://user-images.githubusercontent.com/40893452/44390631-bbc17600-a568-11e8-97e0-92fdede287b9.png)가 존재합니다. 
4. action ![](https://user-images.githubusercontent.com/40893452/44390581-9af92080-a568-11e8-892d-77664e9d06a8.png)는 state와 game score를 변하게 합니다.  
5. ![](https://user-images.githubusercontent.com/40893452/44390499-5ff6ed00-a568-11e8-9259-0c24668359a4.png) 는 stochastic 합니다.  
6. agent는 게임 emulator 내부를 알 지 못하고 오로지 image ![](https://user-images.githubusercontent.com/40893452/44390734-f9260380-a568-11e8-92e3-4a7da2afcdc4.png) 만을 관찰 (observe) 할 수 있습니다. ![](https://user-images.githubusercontent.com/40893452/44390734-f9260380-a568-11e8-92e3-4a7da2afcdc4.png) 는 current screen을 표현하는 raw pixel의 vector입니다.
7. agent는 action sequence의 결과로 reward ![image](https://user-images.githubusercontent.com/40893452/44390851-44401680-a569-11e8-88f3-193afae677d2.png)를 받습니다.  
> 이때, game score는 prior sequence of actions 와 observations 에 의존하며 이에 대한 피드백은 많은 time-step후에 이루어 집니다.  
8. sequence of actions ![](https://user-images.githubusercontent.com/40893452/44390971-92551a00-a569-11e8-9f4f-e22fb77b9551.png) 는 ![image](https://user-images.githubusercontent.com/40893452/44390994-a4cf5380-a569-11e8-8281-968cf85bac7f.png) 로 정의됩니다. 그리고 agent의 게임 전략은 이런 sequence들에 의존하여 학습하게 됩니다.  
9. 모든 ![](https://user-images.githubusercontent.com/40893452/44390971-92551a00-a569-11e8-9f4f-e22fb77b9551.png) 는 finite number of time-step에서 종료 (terminate) 된다고 가정합니다. 그러므로, 크지만 finite Markov decision process (MDP) 로 가정할 수 있게 됩니다.  

agent의 목표는 future rewards를 최대화 하는 action들을 선택하는 것이 목표 입니다. future reward들은 discount factor에 의해서 감소되도록 구성하며, 아래와 같이 표현합니다.  
![](https://user-images.githubusercontent.com/40893452/44391252-49ea2c00-a56a-11e8-9038-4554310cd098.png)  
optimal action-value function 은 strategy에서 얻을 수 있는 expected return (기대 이익)을 최대화 하는 함수를 의미합니다.  
![image](https://user-images.githubusercontent.com/40893452/44391351-92094e80-a56a-11e8-99f6-a583c1fc8ee6.png)  

optimal action-value function은 "Bellman equation"을 따릅니다.  
> ![image](https://user-images.githubusercontent.com/40893452/44391486-f75d3f80-a56a-11e8-811f-24440eee9c2f.png)은 sequence s'에서 모든 가능한 a'에 대한 optimal value가 알려져 있다면, optimal strategy 는 action a'를 선택하는 것이며; 이것은 expected value ![](https://user-images.githubusercontent.com/40893452/44391568-2673b100-a56b-11e8-98c9-5357ae980352.png) 를 최대화 하는 것입니다.  
그러므로, 아래와 같이 정의할 수 있습니다.  
![](https://user-images.githubusercontent.com/40893452/44391597-3db29e80-a56b-11e8-9dfc-6d0110f98c73.png)  

많은 RL algorithm들의 기본적인 아이디어는 action-value function을 "Bellman equation"을 통해서 평가 (estimate) 하는 것입니다. 이때 "Iterative update" 방식이 사용되며 다음과 같이 표현됩니다.  
![image](https://user-images.githubusercontent.com/40893452/44391893-ed880c00-a56b-11e8-81d6-9568a6db7e45.png)  
> i+1 step에서의 action value function은 i step 에서의 s와 a 가 알려졌을때의 action에 대한 기댓 값으로 업데이트 됩니다.  
그러나, 이런 방법은 여러 sequence가 존재할 때 모든 sequence 마다 평가 (estimation)을 수행해야 한다는 점에서 "action value function"은 비 실용적입니다.  

대신에, action-value function을 평가하기 위해서 "function approximator"를 사용하는 것이 일반적인 방법입니다. 이를 아래와 같이 표현할 수 있습니다.  
![](https://user-images.githubusercontent.com/40893452/44392347-f4634e80-a56c-11e8-901e-04c33b3f125c.png)  
강화학습 커뮤니티에서는 전통적으로 linear approximator를 사용하였지만 최근에는 neural network 와 같은 non-linear function approximator가 사용되고 있습니다. 

Q-network는 iteration i 마다 변화하는 loss function ![image](https://user-images.githubusercontent.com/40893452/44392470-368c9000-a56d-11e8-9329-e662afe2176a.png)의 sequence 를 최소화 하는 방향으로 학습이 진행됩니다.  
![image](https://user-images.githubusercontent.com/40893452/44392493-4b692380-a56d-11e8-9fc5-029ae520b619.png)  
이때, ![image](https://user-images.githubusercontent.com/40893452/44392538-68055b80-a56d-11e8-88af-c2eb203c7b2e.png)는 iteration i 에서의 target을 의미하며, p(s,a)는 "behavior distribution"이라고 불리는 action a가 sequence s에서 발생할 probability distribution 입니다.  

이전 iteration i-1에서 parameter들은 training 과정에서 고정됩니다. target은 network weight에 의해서 영향을 받는 다는것에 주목해야하며, supervised learning 방법과 유사한 개념을 만드는 중요한 요소입니다. 학습을 위한 미분 값은 아래와 같이 표현됩니다.  
![image](https://user-images.githubusercontent.com/40893452/44392782-fbd72780-a56d-11e8-8abd-711bcce51c3b.png)  

이 논문에서 제시하는 알고리즘은 다음과 같은 특징을 가집니다.  
1. model-free : 이 모델은 환경정보로부터 직접적으로 task를 해결하는 RL 입니다.  
2. off-policy : greedy strategy 에 대해서 학습하며, state space의 충분한 탐색을 보장하는 "behavior distribution"을 따라서 학습한다고 가정합니다.  

## Deep Reinforcement Learning
Tesauro's TD-Gammon architecture가 이 논문에서 제안하는 구조의 시작 점이 되었습니다. TD-Gammon은 value function을 평가하는 network의 parameter들을 업데이트하며, on-policy sample를 사용하여 학습합니다.  
TD-Gammon과 on-policy 와 다르게, 이 논문에서 제시하는 알고리즘은 "experience replay"라 불리는 방법을 사용합니다.   
episode라 불리는 데이터 e(t) = (s(t), a(t), r(t), s(t+1)) 를 agent의 experience로써 "replay memory"에 저장해둡니다.   
Q-learning의 update (minibatch update) 시에 "replay memory"에서 random하게 episode를 가져와서 학습을 수행합니다.  
Sampling된 episode를 기반으로 "e-greedy policy"를 적용합니다.  
위의 deep q-learning 알고리즘은 몇가지 장점을 가집니다.  
1. 각각의 experience는 여러번 학습과정에서 사용됩니다. 이는 데이터의 효율성을 증가시킵니다.    
2. 연속적으로 발생하는 experience sample들만으로 학습을 하는것은 비효율 적입니다. 이는 연속으로 발생하는 sample들 간의 "Correlation"이 높기 때문입니다. randomizing sample 방법을 통해서 이러한 experience data간의 correlation을 학습 과정에서 줄이고 업데이트의 variance를 줄입니다.  
> 연속된 sample은 유사한 데이터만 발생되기 때문..  
3. on-policy 를 따라서 학습이 이루어질 때, current parameter들은 학습이 이루어진 후에 next data sample들을 결정하게 됩니다.  
> 예를들어, maximizing action이 left인 경우 training sample은 left-action에 의해 나오는 결과들로 주로 이루어질 것입니다. 만약 action이 right로 중간에 변한다면, 그때 training sample distribution 역시 변하게 될 것입니다. 이는 원하지 않는 feedback loop를 증가시킵니다. 그리고 local minimum과 같은 문제를 발생시킵니다.  

그러므로, "replay memory"를 통한 학습 방법을 통해서, "behavior distribution"이 이전의 state를 기준으로 평균적으로 생성하게 되어 학습 과정을 smooth하게 만듭니다. 그리고, parameter들의 발산과 학습 과정에서의 잦은 변화를 줄입니다.  

Off-policy에서 "experience replay"가 이루어지며, 이는 sample을 생산해내는 parameter와 current parameter가 다른 것을 의미합니다.  

## Preprocessing and Model Architecture
기본 Atari game은 210x160 pixel image 이며 128 color palette에서 이루어집니다. 
이 논문에서는 input dimensionality를 줄이기 위해서 preprocessing을 진행합니다.  
"Raw frame"들은 우선 RGB 표현을 "Gray-scale"로 변환합니다. 그리고 "down-sampling"으로 110x84 pixel image로 변환합니다.  
그리고 최종적으로 input은 84x84의 playing area를 잘라낸 형태로 정해집니다.  
> 이는 CNN 2D 구조를 활용하므로 input의 square 형태에 효과적인 적용을 위해서 84x84 형태로 input을 변형하였다고 합니다.  
![image](https://user-images.githubusercontent.com/40893452/44437402-202a1700-a5f5-11e8-8b71-133c30c6e8a3.png)  

Algorithm 1의 phi function은 게임의 최근 "4 frame"을 위의 preprocessing을 적용합니다. 그리고 Q-function의 input으로 만들기 위해서 변환된 "4 frame"들을 쌓아 버립니다 (stacked).  

model은 separate output unit (각각의 action에 대응) 을 가지게 하며, 단지 state 정보만이 neural network의 input으로 들어가도록 합니다. 그러므로, neural network의 output은 input state에 대응되는 action들에 대응되는 "predicted Q-value"가 됩니다.  
이 구조의 장점은, forward pass 가 단지 한번 (single) 로 이루어지는 것만으로도 모든 action에 대해서 Q-value를 계산할 수 있다는 것입니다.  

## Model Structure
1. input data : 84x84x4 image  
2. convolve 16 8x8 filters with stride 4 , ReLU를 activation function으로 사용  
3. convolve 32 4x4 filters with stride 2 , ReLU 사용  
4. 256 개의 unit으로 이루어진 fully-connected layer , ReLU 사용  
5. output layer는 action 수만큼의 unit의 output을 가진 fully-connected layer  

## Optimizer
Adaptive learning rate method인 RMSProp method를 optimizer로 사용하였습니다.  

## Training and Stability
RL에서 agent의 training의 진척도를 평가하는 것 또한 어려운 문제입니다.  
1. evaluation metric으로써 "total reward"를 사용하며, episode동안 agent가 얻은 reward를 의미합니다.  
average total reward metric은 매우 noisy한 형태를 보입니다. 이유는 neural network의 parameter가 조금의 변화를 보여도 policy에 따라서 방문하게 되는 state의 distribution이 매우 크게 변할 수 있기 때문입니다.  
2. "policy's estimated action-value function Q"를 evaluation metric으로 사용합니다.  
이 값은 주어진 state에서 policy를 따랐을 때 얻게되는 agent의 discounted reward의 추정치를 의미합니다. 이 값은 "total reward"보다는 안정적인 변화를 보여줍니다.  
![image](https://user-images.githubusercontent.com/40893452/44437418-320bba00-a5f5-11e8-876a-8215b6b13ac2.png)  


## Review
Deep Mind 팀에서 발표한 DQN 논문입니다. 이 논문에서는 replay memory를 통해서 학습 과정에서의 experience sample들의 correlation을 제거하여 학습을 효과적으로 편향성을 줄이면서 학습할 수 있는 방법을 제안 하였습니다. 그 뿐 아니라, input으로 state들 만이 주어질 수 있도록 하는 neural network가 적용되는 강화학습 알고리즘을 제안하므로써 새로운 Reinforcement learning architecture의 가능성을 제시합니다.



















