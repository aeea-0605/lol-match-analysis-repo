# 리그오브레전드 유저 분류 프로젝트
---

## 1. 개요
<br/>

### **1-1. 문제 정의**
리그 오브 레전드를 플레이함에 있어 오브젝트의 선점과 플레이어 개인 능력에 따라 승패가 좌우된다. 하지만 오브젝트 선점 여부 및 플레이어 개인 능력에 따른 승률, 라인별 챔피언 픽률 등의 통계치에 대한 정보 제공이 부족하다.

<br/>

### **1-2. 프로젝트 목표**
1. 오브젝트 및 플레이어 개인 능력 관점에서의 분석을 통해 소환사를 분류할 수 있는 기준을 제시한다.
2. 각 기준에 대한 모델을 생성해 Feature Importance 및 특성 파악, 승률이 높은 포지션 및 챔피언에 대한 정보를 제공한다.
3. 해당 프로젝트를 통해 유저들에게 분석 결과를 제공하며, 타 통계사이트의 서비스와 연계할 수 있는 방향을 제시한다.

<br/>

### **1-3. 기술적 Summary**
- numpy, pandas, scikit-learn 를 사용한 데이터 전처리 및 세분화
- PCA, T-SNE 를 통한 유저 데이터 차원축소
- KMeans, HDBSCAN Clustering 을 통한 유저 군집 분석
- Elbow & Silhouette method 및 시각화를 통한 군집 성능 평가
- OLS Logit 모델의 통계적 분석을 통한 Feature Importance 평가

<br/>

### **1-4. 분석 기준 및 모델의 서비스 목표**
#### **<모델 1>**
*오브젝트 관점에서 선점 여부에 대한 Feature 를 통해 오브젝트에 대한 중요도 분석과 기여율이 높은 포지션 및 챔피언에 대한 정보 제공 및 추천*
- 가정 : 플레이 시간에 따라 중요 오브젝트가 달라질 것이다.
- 유저를 나눈 기준 : 플레이 시간 (gameduration)
    1. 20분 미만
    2. 20 ~ 30
    3. 30 ~ 40
    4. 40분 초과

#### **<모델 2>**
*유저 개인 능력 관점에서 스텟 Feature 를 통해 유저를 대표할 수 있는 군집 형성 및 특성 분석, 각 군집의 개선점 제공과 군집 내 승률이 높은 챔피언에 대한 정보 제공 및 추천*
- 가정 : 각 포지션마다 유저를 대표할 수 있는 군집들이 존재할 것이다.
- 유저를 나눈 기준
    - 포지션 (MID, TOP, JUNGLE, SUPPORT, ADC)
    - Unsupervised Clustering 을 통한 군집 형성

<br/>

### **1-5. 데이터셋 및 설명**
#### **raw data**
- lol 10.18 버전 KR 지역 솔로랭크 match data (약 17만건)
- KR 지역 솔로랭크 다이어몬드 티어 이상의 소환사가 1명 이상 포함된 경기
- 단일 경기의 기록이 한 줄에 들어가 있는 JSON 타입의 데이터

#### **Preprocessed Dataset for Analysis**
1. game_df : raw data가 포함된 데이터셋 (dataframe)
    - 168931 rows
2. champion_datas : 챔피언에 대한 key, name이 있는 데이터 (json)
    - 총 156개의 챔피언에 대한 정보
3. participant_df : 유저의 개인능력 data가 포함된 데이터셋 (dataframe)
    - Feature : game_df의 'participants' Column
    - row : game_df의 1row(한 경기) 를 10rows(10명의 유저) 로 세분화 > 1698280 rows
4. participantextendedstat_df : 유저의 라인 및 티어 data가 포함된 데이터셋 (dataframe)
    - Feature : game_df의 'participantextendedstats' Column
    - row : game_df의 1row를 10rows로 세분화 > 1698280 rows
5. team_df : 유저의 팀 및 오브젝트 관여여부 data가 포함된 데이터셋 (dataframe)
    - Feature : game_df의 'teams', 'gameduration' Columns
    - row : game_df의 1row를 10rows로 세분화 > 1698280 rows
        - 0 ~ 4 유저 : 팀 100
        - 4 ~ 9 유저 : 팀 200
6. team_df_2 : 각 게임의 두 팀에 대한 data가 포함된 데이터셋 (dataframe)
    - Feature : game_df의 'teams', 'gameduration' Columns
    - row : game_df의 1row를 2rows(두 팀)로 세분화 > 339662 rows
7. la_mid_df : MID로 세분화된 data와 군집화에 대한 label data가 포함된 데이터셋(dataframe)
    - Feature : feature selection을 통한 23개의 변수추출
    - row : 335822 rows (미드 유저들에 대한 335822개의 경기 데이터)

<br/>

---
---
## 2. 결론

### **2-1. <모델 1>에 대한 분석 결과**

#### **2-1-1. 플레이 시간에 따른 오브젝트 선점 중요도**
- 분석에 사용한 데이터셋 : team_df_2

<br/>
<img width="758" alt="2-1-1" src="https://user-images.githubusercontent.com/80459520/133557047-7a2b3fdd-1e25-4d86-aa9a-490d0cc69a1c.png">

- 20분 미만 : 억제기(100%) > 타워(96%) > 전령(83%) > 드래곤(78%) > 킬(72%)
- 20 ~ 30 : 억제기(99%) > 바론(94%) > 타워(75%) > 드래곤(64%) ≥ 전령(64%) > 킬(60%)
- 30 ~ 40 : 억제기(86%) > 바론(74%) > 타워(57%) > 드래곤(54%) ≥ 전령(54%) > 킬(53%)
- 40분 초과 : 억제기(60%) > 바론(53%) ≥ 드래곤(53%) > 킬(51%) ≥ 타워(51%) > 전령(50%)
```
rate(%) = 특정 오브젝트를 선점하고 이긴 팀 수(win) / 특정 오브젝트를 선점한 총 팀 수(total)

<해석 Example>
플레이 시간이 20 ~ 30일 때, 첫 타워 선점여부의 중요성은 3위이며 해당 오브젝트를 선점했을 때 승률은 75% 이다.
```

> 2-1-1 인사이트
- 가장 중요한 오브젝트는 억제기이며, 기준에 따라 중요도순이 달라진다.
- 플레이 시간이 증가함에 따라 오브젝트 선점여부에 관한 중요도는 점점 감소하며. 유저의 개인 능력에 따라 승패가 좌요된다는 것을 알 수 있다.

<br/>

---

#### **2-1-2. 각 기준에서 오브젝트에 대해 가장 많이 기여한 포지션 및 TOP2  챔피언**
- 분석에 사용한 데이터셋 : participant_df, participantextendedstat_df, team_df, champion_datas
- 총 23개의 결과 중 2개를 샘플링하여 결과를 제공하였습니다.

**< Sample 1 > : 20분 미만, 첫 억제기 파괴에 관여**

<img width="500" alt="2-1-2_sample1" src="https://user-images.githubusercontent.com/80459520/133558650-c0274e97-f08a-4ea8-a5ab-f31df6c563da.png">

- Rank 1. 서포터 (22.75%)
    - TOP2 Champion : 쓰레쉬(10.49%), 판테온(7.2%)
- Rank 2. 원딜 (21.72%)
    - TOP2 Champion : 케이틀린(14.56%), 이즈리얼(14.24%)
- Rank 3. 정글 (19.85%)
    - TOP2 Champion : 니달리(10.02%), 리신(9.55%)
- Rank 4. 미드 (18.59%)
    - TOP2 Champion : 아칼리(6.94%), 사일러스(6.05%)
- Rank 5. 탑 (17.09%)
    - TOP2 Champion : 카밀(6.31%), 루시안(6.25%)
```
<해석 Example>
플레이 시간이 20분 미만일 때, 첫 억제기를 파괴에 관여하여 이긴 유저의 포지션 중 서포터가 22.75%로 가장 많았고,
그 중 쓰레쉬와, 판테온이 각각 10.49%, 7.2%로 비율이 가장 높다.
```

**< Sample 2 > : 20~30, 첫 타워 파괴에 관여**

<img width="500" alt="2-1-2_sample2" src="https://user-images.githubusercontent.com/80459520/133559885-a9791375-7b96-417f-b771-65c9d97a9a90.png">

- Rank 1. 정글 (23.34%)
    - TOP2 Champion : 리신(9.77%), 그레이브즈(9.5%)
- Rank 2. 탑 (23.23%)
    - TOP2 Champion : 레넥톤(8.12%), 카밀(6.71%)
- Rank 3. 원딜 (21.71%)
    - TOP2 Champion : 케이틀린(16.74%), 이즈리얼(15.25%)
- Rank 4. 서포터 (17.5%)
    - TOP2 Champion : 쓰레쉬(9.33%), 룰루(7.74%)
- Rank 5. 미드 (14.21%)
    - TOP2 Champion : 루시안(7.09%), 제드(5.3%)
```
<해석 Example>
플레이 시간이 20 ~ 30일 때, 첫 타워 파괴에 관여하여 이긴 유저의 포지션 중 정글이 23.34%로 가장 많았고,
그 중 리신, 그레이브즈가 각각 9.77%, 9.5%로 비율이 가장 높다.
```
> 2-1-2 인사이트
- 플레이 시간대에 각각의 오브젝트를 선점하고 이긴 유저들의 데이터를 통해 포지션별 관여율과 각 포지션에서 픽률이 높은 챔피언에 대한 정보를 제공해줄 수 있다.

<br/>

---
---

### **2-2. <모델 2>에 대한 분석 결과**
- 분석에 사용한 데이터셋 : participant_df, participantextendedstat_df, team_df, la_mid_df
- *5개의 포지션 중 MID를 선정하여 모델링을 진행하였습니다.*

<br/>

#### **2-2-1. 최종 분류 모델 선정 및 군집 특성 파악**
- **최종 모델**
    - Input Data
    > MinMax Scaling → PCA(dims: 20 > 10) → T-SNE(dims: 10 > 2, perplexity=50)
    - Classifier
    > HDBSCAN(min_cluster_size=5, gen_min_span_tree=True, min_samples=200, cluster_selection_epsilon=1.6, prediction_data=True)

**<최종 모델에 대한 Clustering Visualization>**

<img width="775" alt="2-2-1_1" src="https://user-images.githubusercontent.com/80459520/133562640-988608e8-68ba-42b6-bab5-82121f7a2042.png">

- Noise datas(label= -1)를 제외했을 때, 총 0 ~ 7 까지의 8개의 군집 형성

**<군집별 승리 빈도 및 승률>**

<img width="700" alt="2-2-1_2" src="https://user-images.githubusercontent.com/80459520/133563099-a1ea26df-2aa0-4379-b0df-014ee4415d78.png">

- 3(99.91%) > 0(99.64%) > 5(98.92%) > 4(93.51%) > 2(74.7%) > 1(50.37%) > 7(42.78%) > 6(9.04%)

**<군집별 Feature Visualization>**

<img width="721" alt="스크린샷 2021-09-16 오후 3 48 27" src="https://user-images.githubusercontent.com/80459520/133563893-a97e3cbe-cd25-4d70-8d01-628525edc748.png">

**<각 Feature에 대한 BEST, WORST TOP2 군집 선정>**

<img width="1022" alt="스크린샷 2021-09-16 오후 4 04 06" src="https://user-images.githubusercontent.com/80459520/133565740-127b9e23-2496-4439-952e-1270371d7cba.png">

<br/>

---

#### **2-2-2. 군집별 픽률 TOP3 챔피언**
![2-2-2_1](https://user-images.githubusercontent.com/80459520/133569094-0da0fbc6-e922-4f3c-a543-4ebba3568ab9.png)

<br/>

> 2-2 인사이트

포지션 내에서도 각자의 특징을 가지고 있는 군집들이 형성되어 있음을 알 수 있고, 각 군집에 대한 플레이 성향 및 승률, 픽률 높은 챔피언에 대한 정보를 줄 수 있다.

1. Cluster_0
    - BEST 1: 적 정글 중립미니언킬, 오브젝트에 가한 피해량, 타워킬, 골드획득, 총 중립미니언킬, 포탑에 가한 피해량, CS수
    - BEST 2: 총 가한 군중제어 시간
    - WORST 1: -
    - WORST 2: 와드 제거 수, 시야점수, 적에게 가한 CC시간
2. Cluster_1
    - BEST 1: 총 가한 군중제어 시간, 받은피해량, 적에게 가한 CC시간
    - BEST 2: 와드 제거 수, 시야점수, 팀 정글 중립미니언킬, 감소 피해량, 힐량
    - WORST 1: -
    - WORST 2: -
3. Cluster_2
    - BEST 1: -
    - BEST 2: 와드 설치 수, 힐 횟수
    - WORST 1: 감소 피해량, 받은 피해량
    - WORST 2: 오브젝트에 가한 피해량, 총 가한 군중제어 시간, 힐량, 챔피언에게 가한 피해량, 최대 생존 시간
4. Cluster_3
    - BEST 1: 타워킬, 억제기킬
    - BEST 2: 골드획득, 포탑에 가한 피해량, 오브젝트에 가한 피해량
    - WORST 1: -
    - WORST 2: 받은 피해량
5. Cluster_4
    - BEST 1: 팀 정글 중립미니언킬, 와드 제거 수, 시야점수, 감소 피해량, 힐량, 챔피언에게 가한 피해량
    - BEST 2: 총 중립미니언킬, 적 정글 중립미니언킬, 억제기킬, 받은 피해량, 최대 생존 시간
    - WORST 1: 힐 횟수
    - WORST 2: -
6. Cluster_5
    - BEST 1: 와드 설치 수
    - BEST 2: CS수
    - WORST 1: -
    - WORST 2: -
7. Cluster_6
    - BEST 1: 힐 횟수, 최대 생존 시간
    - BEST 2: -
    - WORST 1: 오브젝트에 가한 피해량, 총 가한 군중제어 시간, 와드 제거 수, 시야점수, 와드 설치 수, 적에게 가한 CC시간, 챔피언에게 가한 피해량, CS수, 힐량, 포탑에 가한 피해량, 골드획득, 총 중립미니언킬
    - WORST 2: 감소 피해량
8. Cluster_7
    - BEST 1: -
    - BEST 2: 챔피언에게 가한 피해량
    - WORST 1: - 
    - WORST 2: 골드획득, 포탑에 가한 피해량, CS수

<br/>

---
---

### **2-3. OLS Logit Model을 통한 Feature Importance 분석**
- 분석에 사용한 데이터셋 : la_mid_df
    - 독립변수 : win, label, championid를 제외한 변수
    - 종속변수 : win
- 2-2와 동일하게 MID 데이터셋에 대해 모델링을 진행하였습니다.

**<Best OLS Logic Model에 대한 Summary>**

![2-3-1](https://user-images.githubusercontent.com/80459520/133571058-4e45d281-17b9-46fe-8468-83a777d1db21.png)

- 해당 모델에 대한 5번의 교차 검증 Accuracy : 84.6%
- 20개의 독립변수들 모두 p-value가 0에 수렴하므로, 모든 독립변수가 종속변수(Win)에 영향을 준다고 할 수 있다.

**< Feature Importance Visualization >**

![2-3-2](https://user-images.githubusercontent.com/80459520/133572730-5231b88d-2fb9-4fc7-a3a8-4fa78c7ce5a4.png)

- <한 단위 증가함에 따라 승리할 확률이 높아지는 독립변수들의 중요도 순위>
    - 골드획득 > 타워에 가한 피해량 > 힐량 > 억제기킬 > 적 정글 중립미니언킬 > 시야점수 > 오브젝트에 가한 피해량 > 총 가한 군중제어 시간 > 타워킬 > 감소 피해량
- <한 단위 증가함에 따라 패배할 확률이 높아지는 독립변수들의 중요도 순위>
    - 받은 피해량 > CS수 > 와드 제거 수 > 챔피언에게 가한 피해량 > 총 중립미니언킬 > 팀 정글 중립미니언킬 > 와드 설치 수 > 적에게 가한 CC시간 > 힐 횟수 > 최대 생존 시간

<br/>

> 2-3 인사이트
- 각 포지션별로 어떤 독립변수가 승/패에 얼마나 영향을 주는지에 대한 정보를 제공할 수 있다.
- 2-2에서의 군집별 특성을 파악하고, 2-3의 Feature Importance를 고려하여 각 군집별로 어떤 부분을 먼저 개선해야할지에 대한 방향을 제시할 수 있다.

<br/>

---
---
## 3. 활용 및 서비스 제시
- 유저가 미드를 주 포지션으로 게임을 플레이한다고 가정
    - 특정 유저에 대한 10개의 전적 데이터를 Input data로 설정

<br/>

#### **0. 랜덤샘플링을 위한 10개의 index 추출**

<img width="600" alt="스크린샷 2021-09-16 오후 10 39 48" src="https://user-images.githubusercontent.com/80459520/133622622-0a734a9e-0923-4c1c-8cf8-f431b34530ab.png">

- 주 포지션이 미드라는 가정하에 MID 데이터셋에서 랜덤으로 10개의 index 추출

<br/>

### **<모델 1>**
#### **1. MID 데이터셋에서 랜덤샘플링 및 표본평균 추출**

<img width="600" alt="스크린샷 2021-09-16 오후 10 46 03" src="https://user-images.githubusercontent.com/80459520/133623606-4948a765-9d87-4b98-969b-14bf2d283375.png">

- 해당 유저의 평균 플레이 시간은 26.47분이다.

#### **2. 표본평균을 통한 기준 플레이 시간 정의**

<img width="600" alt="스크린샷 2021-09-16 오후 10 47 44" src="https://user-images.githubusercontent.com/80459520/133623855-a3518acb-0bd3-44b4-b2f4-3c05fc5618e7.png">

- 해당 유저가 속하는 기준 플레이 시간은 1200(20) ~ 1800(30) 이다.

#### **3. 유저의 기준 플레이 시간에 대한 오브젝트 중요도 정보 제공**

<img width="403" alt="스크린샷 2021-09-16 오후 10 50 12" src="https://user-images.githubusercontent.com/80459520/133624203-ebe2c519-ae4b-429c-8649-aac247f41725.png">

- 유저가 속한 기준에서의 오브젝트 중요도 순서 및 승률에 대한 Visualization

#### **4. 유저의 기준 플레이 시간에서 특정 오브젝트에 관여하고 이긴 포지션 및 TOP2 챔피언 정보 제공**

<img width="616" alt="스크린샷 2021-09-16 오후 10 52 09" src="https://user-images.githubusercontent.com/80459520/133624524-67a35f93-1fef-4b1b-840e-c5f20edb3f8c.png">

- 4번에서의 6가지의 오브젝트 중 유저가 첫 억제기 파괴 관여에 대한 Feature를 선택했다고 가정
```
<결과>
유저의 전적 데이터 분석 결과, 평균 플레이 시간은 20분 ~ 30분 사이입니다.

해당 플레이 시간에서의 오브젝트 중요도는 억제기(99%) > 바론(94%)
> 타워(75%) > 드래곤(64%) > 전령(64%) > 킬(60%) 입니다.

MID 포지션은 첫 억제기 파괴에 관여에 대한 기여도가 4위(18.62%)이며,
그 중 사일러스(5.63%), 아칼리(5.54%)가 가장 많이 픽이 되었습니다.
가장 많이 기여한 포지션은 원딜(22.06%)입니다.
```
<br/>

### **<모델 2>**
#### **1. T-SNE 데이터셋에서 랜덤샘플링 및 학습된 HDBSCAN 모델을 통한 군집예측**

<img width="636" alt="스크린샷 2021-09-16 오후 11 10 30" src="https://user-images.githubusercontent.com/80459520/133627522-608a755d-46ea-47a3-9560-29fadce681b9.png">

- approximate_predict method를 통해 표본들에 대한 label을 부여

#### **2. predicted label 및 해당 label의 출현비율에 대한 데이터로 정리**

<img width="539" alt="스크린샷 2021-09-16 오후 11 29 51" src="https://user-images.githubusercontent.com/80459520/133630686-ead47ee0-98d0-4f88-b2ce-e5c13ced0397.png">

- Cluster_0 : 10%, Cluster_6 : 10%, Cluster_7 : 80% 의 비율을 가짐

#### **3. 출현비율과 군집의 weight를 통한 Score 계산**

<img width="729" alt="스크린샷 2021-09-16 오후 11 35 38" src="https://user-images.githubusercontent.com/80459520/133631724-a142dc32-dbb2-4619-9368-899c5ba1bd00.png">

- Cluster_0 score : 24.44, Cluster_6 score : 9.26, Cluster_7 score : 0.96
- 군집의 weight(label_score) = 각 Cluster의 count / total_count
- score = 출현비율 / 군집의 weight

#### **4. 유저를 대표하는 군집 선정**

<img width="662" alt="스크린샷 2021-09-16 오후 11 46 05" src="https://user-images.githubusercontent.com/80459520/133633565-c20c7e9d-379b-4160-84d3-86212a9e096a.png">

- 3번에서 가장 높은 score를 보유한 군집을 유저를 대표하는 군집으로 선정
- 해당 유저를 대표하는 군집은 Cluster_0 이다.

#### **5. 해당 군집의 TOP3 챔피언 정보 제공**

<img width="597" alt="스크린샷 2021-09-16 오후 11 47 56" src="https://user-images.githubusercontent.com/80459520/133633862-9d4b4efd-d308-4bc7-b484-b09caf651017.png">

- Cluster_0 에서 가장 픽률이 높은 TOP3 챔피언은 야스오, 요네, 탈론 이다.
```
<결과>
유저의 전적 데이터 분석 결과, 유저가 속한 군집은 Cluster_0 입니다.

해당 군집은 다른 군집에 비해 미니언 및 오브젝트 관여율이 높고 골드획득량이 높고
와드 제거, 시야점수, 적에게 가한 CC시간에 대한 관여율이 낮습니다.

관여율이 낮은 변수 중 시야점수의 중요도가 가장 높기에 시야점수를 관리한다면 게임에서 이길 확률이 높아질 것입니다.

해당 군집에서 높은 픽률을 보이는 챔피언은 야스오(9.55%), 요네(8.53%), 탈론(6.92%) 입니다.
```

<br/>

---
---

## 4. 과정

#### **상세한 분석 과정은 Notion으로 작성하였습니다.**
[Notion 바로가기](https://puzzled-plantain-00d.notion.site/da2646de4ad2489196eaae008ea41ff2)

<br/>

---
---

## 💡 제언 및 한계점
- 리그오브레전드에 좀 더 전문적인 도메인 지식을 보유하고 Feature Selection을 하고 차원축소, 군집화, 회귀 모델을 생성했다면 좀 더 신뢰성 있고 유저에게 최적화된 인사이트를 도출할 수 있다고 생각합니다.

- Ensemble 모델을 사용해 성능을 높이고, Shap-value 로 Feature Importance 및 영향력을 평가하는 것도 설명력을 좀 더 높일 수 있다고 생각합니다.

<br/>

---
---
## Code Explanation

- [preprocessing.ipynb](https://github.com/aeea-0605/lol-match-analysis-repo/blob/main/preprocessing.ipynb)
    - 데이터에 대한 Documentation
    -  데이터 로드, 전처리, 저장에 대한 Notebook
- [team_object_analysis.ipynb](https://github.com/aeea-0605/lol-match-analysis-repo/blob/main/team_object_analysis.ipynb)
    - <모델_1> 의 분석을 위한 데이터 전처리 및 세분화
    - <모델_1> 의 오브젝트에 대한 중요도 분석에 대한 Notebook
- [user_cluster_analysis.ipynb](https://github.com/aeea-0605/lol-match-analysis-repo/blob/main/user_cluster_analysis.ipynb)
    - <모델_2> 의 분석을 위한 데이터 전처리 및 세분화
    - <모델)2> 의 MID 데이터셋의 최종 군집 분석 모델에 대한 특성파악 및 회귀 분석에 대한 Notebook
- [colab_mid_col33_cluster.ipynb](https://github.com/aeea-0605/lol-match-analysis-repo/blob/main/colab_mid_col33_cluster.ipynb)
    - Colab
    - MID 데이터셋에 대한 1차 군집 분석 및 성능 평가
        - 독립변수의 개수 : 33개
        - Clusterer : KMeans
        - 차원 축소 : PCA
        - 성능 평가 방법 : Elbow, Silhouette method
- [colab_mid_col20_cluster.ipynb](https://github.com/aeea-0605/lol-match-analysis-repo/blob/main/colab_mid_col20_cluster.ipynb)
    - Colab
    - MID 데이터셋에 대한 2차 군집 분석 및 성능 평가
        - 독립변수의 개수 : 20개
        - Clusterer : KMeans
        - 차원축소 : PCA
        - 성능 평가 방법 : Elbow, Silhouette method
- [colab_tsne_tuning.ipynb](https://github.com/aeea-0605/lol-match-analysis-repo/blob/main/colab_tsne_tuning.ipynb)
    - Colab
    - T-SNE 하이퍼 파라미터 튜닝을 통한 MID 데이터셋의 3차 군집 형성
        - 독립변수의 개수 : 33, 20
        - 차원축소 : PCA, T-SNE
        - 성능 평가 방법 : Visualization
- [colab_hdbscan_tuning.ipynb](https://github.com/aeea-0605/lol-match-analysis-repo/blob/main/colab_hdbscan_tuning.ipynb)
    - Colab
    - HDBSCAN 하이퍼 파라미터 튜닝을 통한 최종 군집 모델 생성
- [newdata_prediction.ipynb](https://github.com/aeea-0605/lol-match-analysis-repo/blob/main/newdata_prediction.ipynb)
    - 새로운 Input data에 대한 해당 프로젝트에 대한 결과 도출 및 정보 제공
- [module.py](https://github.com/aeea-0605/lol-match-analysis-repo/blob/main/module/module.py)
    - 분석에 사용한 함수 및 변수가 저장된 모듈 파일
- [hdbscan_obj.pkl](https://github.com/aeea-0605/lol-match-analysis-repo/blob/main/models/hdbscan_obj.pkl)
    - 최종 HDBSCAN 모델에 대한 객체 pickle 파일
