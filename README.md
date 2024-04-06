# Relation Extraction(RE)

## 프로젝트 개요
### 대회 소개
<img width="750" src="https://github.com/rlarlgh96/relation-extraction/assets/121072239/6497ff3c-95aa-4ef8-812e-3f0b2c1239c4"><br>
- 본 프로젝트는 네이버 부스트캠프 AI Tech 6기 NLP 트랙 과정에서 진행한 교육용 대회 프로젝트이다.
- Relation Extraction(RE)는 개체(entity)에 대한 속성과 관계를 예측하는 NLP task로, 본 대회에서는 Bert 모델을 사용해 문장 내 존재하는 두 개체의 관계를 총 30가지 중 하나로 분류한다.

### 평가 방법
- 본 대회에서는 두 가지 평가 지표(no_relation을 제외한 micro F1 score, 모든 label에 대한 area under the precision-recall curve(AUPRC), micro F1 score를 우선시 함)를 사용해 모델의 성능을 평가한다.

## 프로젝트 수행 과정
### Data description
- ***KLUE: Korean Language Understanding Evaluation*** 논문을 읽고 각 label과 feature가 나타내는 의미를 파악하였다.
- label에 따라 가능한 entity type을 파악하였다(예시: org:dissolved는 조직이 해산된 날짜를 나타내므로 가능한 subject type과 object type은 각각 ORG와 DAT).

### EDA
- train 데이터셋에서 label 분포를 확인한 결과, 다음과 같이 불균형한 분포를 보였다.
<img width="1000" src="https://github.com/rlarlgh96/relation-extraction/assets/121072239/e2d60d44-21ae-495e-a881-aa97264f92a7"><br>
- train과 test 데이터셋에서 subject type 분포는 다음과 같다.
<img width="1000" src="https://github.com/rlarlgh96/relation-extraction/assets/121072239/0bdd138e-251b-4fef-8108-747d855bbbb5"><br>
- train과 test 데이터셋에서 object type 분포는 다음과 같다.
<img width="1000" src="https://github.com/rlarlgh96/relation-extraction/assets/121072239/b8e8e9c5-a80b-454a-8535-ccd6ed283d9d"><br>
- entity type 쌍을 나타내는 entity_pair feature를 추가하여 train과 test 데이터셋에서 분포를 확인하였다.
<img width="750" src="https://github.com/rlarlgh96/relation-extraction/assets/121072239/e773b1d4-cf93-4122-b9ec-061848139651"><br>
<img width="750" src="https://github.com/rlarlgh96/relation-extraction/assets/121072239/608a8666-454d-4631-921b-761053bb5ebc"><br>
- train 데이터셋에서 label별 entity type의 갯수를 출력하여 data description 과정에서 추론한 label별 entity type과 일치하는지를 확인하였다.
- 그 결과, 추론한 entity type와 일치하지 않는 데이터가 다수 존재했다. 이를 확인해 본 결과, 일부 단어에서 label에 따라 entity type이 다르게 표기된 것을 확인할 수 있었다(예시: '서울특별시'의 경우, label에 따라 entity type을 ORG, POH, LOC로 표기). 따라서, 추론한 entity type만 가지고 label을 판단하기 어렵다는 생각이 들었다.

### Data preprocessing
- subject_entity와 object_entity에 포함된 정보(word, start_idx, end_idx, entity_type)를 추출하여 이를 각각의 column으로 저장하였다.
- train 데이터셋에서 데이터가 중복되거나 label만 다른 46개의 데이터를 제거하였다.

### Modeling
<img width="1000" src="https://github.com/rlarlgh96/relation-extraction/assets/121072239/b8fe4f53-036e-450f-b588-28be5956edcd"><br>
<img width="1000" src="https://github.com/rlarlgh96/relation-extraction/assets/121072239/a3b53f50-1ddc-47cd-b733-4177fe34bf64"><br>
- ***An Improved Baseline for Sentence-level Relation Extraction*** 논문에서는, sentence level의 relation extraction task에서 Typed entity marker를 사용했을 때 가장 높은 성능을 보였다고 한다.
- 이러한 방식을 적용하기 위해 모델 tokenizer에 entity type에 따라 24가지 다른 special token을 추가하였고, 그에 따라 모델 embedding size를 수정하였다.
- 모델의 input에 넣을 문장에 논문과 같이 Typed entity marker를 달아주었고, 수행할 task를 나타내는 query 문장을 기존의 문장 앞에 추가하였다.
- 모델의 output에서 CLS 토큰의 embedding과 함께 subject entity와 object entity를 나타내는 Typed entity marker의 embedding을 출력하고, 이를 하나로 결합하는 linear layer를 추가하였다. 이렇게 결합한 embedding을 30개의 클래스로 분류하는 분류기에 넣어 label 예측에 활용하였다.

## 프로젝트 수행 결과
- 프로젝트 수행 결과, 모델의 성능을 7.0123점 향상시켰다.
  
  | model | F1 score |
  |--------|--------|
  | baseline | 71.0283 |
  | Typed entity marker | 78.0406 |

## 참고문헌
- Park, S., et al. (2021). ***KLUE: Korean Language Understanding Evaluation***. 	arXiv:2105.09680v4 [cs.CL].
- Zhou, W., & Chen, M. (2022). ***An Improved Baseline for Sentence-level Relation Extraction***. arXiv:2102.01373v4 [cs.CL].
