# Orthogonal Benchmark Cross-Evaluation Pipeline

외부 공식 벤치마크 테스트셋(DeepPrime Generalization Test Set, PRIDICT2.0 23k Fold 0 Test Set)에 대한 모델 교차 평가(Cross-Evaluation) 파이프라인 디렉토리입니다.

## 1. 개요 및 설계 원칙

1. **데이터 누수(Data Leakage) 완전 배제**:
   - DeepPrime6의 전체 라이브러리(`pe6_full_library.csv`, $N = 18,971$)에 노출된 모든 타겟(`WideTargetSequence`) 및 pegRNA 디자인을 평가에서 완전히 제외합니다.
   - 각 baseline 모델(DeepPrime-base, PRIDICT2.0)의 학습/사전학습 데이터와 겹치는 타겟 또한 제외하여 공정한 Strictly Unseen 환경을 구축합니다.
2. **모듈 분리**:
   - `01_filter_testsets.py`: 데이터 로드, 중복 탐색, 누수 필터링, 메타데이터 매핑 및 필터링 데이터 저장
   - `02_run_inference.py`: GPU 배치 추론(DeepPrime-base 20-앙상블, PRIDICT2.0 공식 모델, DeepPrime6 4종 모델) 및 실측값 대비 상관계수(Spearman $\rho$, Pearson $r$) 산출

---

## 2. 디렉토리 구성

```
scripts/orthogonal_evaluation/
├── 01_filter_testsets.py       # Step 1: 데이터 누수 배제 및 필터링 데이터셋 생성
├── 02_run_inference.py         # Step 2: 6종 모델 GPU 추론 및 상관계수 산출
├── check_overlap.py            # 진단/감사용: 라이브러리 간 타겟/디자인 중복 상세 분석
├── run_pipeline.sh             # Step 1 -> Step 2 원클릭 자동 실행 쉘 스크립트
└── README.md                   # 파이프라인 가이드 문서 (본 파일)
```

---

## 3. 평가 대상 모델 (총 6종)

| 모델 | 체크포인트 / 가중치 | 비고 |
| :--- | :--- | :--- |
| **DeepPrime-base** | `src/models/weights/DP_variant_293T_PE2max_epegRNA_Opti_220428/*.pt` | 공식 20-모델 앙상블 평균 |
| **PRIDICT2.0** | `PRIDICT2/model/pridict2_model.pt` | 공식 릴리즈 Fold 0 모델 |
| **DeepPrime6-pe6a** | `src/models/weights/DeepPrime6-weights/pe6a_mainft.ckpt` | PE6a fine-tuned |
| **DeepPrime6-pe6b** | `src/models/weights/DeepPrime6-weights/pe6b_mainft.ckpt` | PE6b fine-tuned |
| **DeepPrime6-pe6c** | `src/models/weights/DeepPrime6-weights/pe6c_mainft.ckpt` | PE6c fine-tuned |
| **DeepPrime6-pemaxdrnaseh** | `src/models/weights/DeepPrime6-weights/pemaxdrnaseh_mainft.ckpt` | PE2max-dRNaseH fine-tuned |

---

## 4. 실행 방법

프로젝트 루트 또는 본 디렉토리 어디에서나 실행 가능합니다.

### A. 파이프라인 전체 실행 (One-click)
```bash
bash scripts/orthogonal_evaluation/run_pipeline.sh
```

### B. 단계별 수동 실행
```bash
# 1단계: 데이터 필터링
python scripts/orthogonal_evaluation/01_filter_testsets.py

# 2단계: 모델 추론 및 성능 산출
python scripts/orthogonal_evaluation/02_run_inference.py

# (선택) 라이브러리 간 중복 상세 감사
python scripts/orthogonal_evaluation/check_overlap.py
```

---

## 5. 결과 산출물 위치 (`data/orthogonal_testset/`)

- `overlap_exclusion_summary.csv`: 테스트셋별 원본 행 수, 중복 건수, 최종 필터링된 평가 행 수 요약
- `deepprime_test_filtered.csv`: DeepPrime 테스트셋에서 누수 배제된 데이터 ($N = 7,416$)
- `pridict2_test_filtered.csv`: PRIDICT2.0 테스트셋에서 누수 배제된 데이터 ($N = 4,577$)
- `deepprime_test_predictions.csv`: DeepPrime 필터링셋에 대한 모델별 예측값 및 실측값
- `pridict2_test_predictions.csv`: PRIDICT2.0 필터링셋에 대한 모델별 예측값 및 실측값
- `deepprime_test_performance.csv`: DeepPrime 필터링셋 성능 비교표 (Spearman, Pearson, N)
- `pridict2_test_performance.csv`: PRIDICT2.0 필터링셋 성능 비교표 (Spearman, Pearson, N)

---

## 6. 최종 벤치마크 결과 요약 (Strictly Unseen)

### DeepPrime Generalization Test Set ($N = 7,416$)
- **DeepPrime-base**: Spearman $\rho = 0.7186$, Pearson $r = 0.5533$
- **DeepPrime6-pemaxdrnaseh**: Spearman $\rho = 0.5811$, Pearson $r = 0.5287$
- **DeepPrime6-pe6b**: Spearman $\rho = 0.5616$, Pearson $r = 0.5026$
- **DeepPrime6-pe6a**: Spearman $\rho = 0.5294$, Pearson $r = 0.4758$
- **DeepPrime6-pe6c**: Spearman $\rho = 0.4871$, Pearson $r = 0.4072$
- **PRIDICT2.0**: Spearman $\rho = 0.0429$, Pearson $r = 0.0268$

### PRIDICT2.0 Official Fold 0 Test Set ($N = 4,577$)
- **PRIDICT2.0**: Spearman $\rho = 0.8915$, Pearson $r = 0.8867$
- **DeepPrime6-pe6a**: Spearman $\rho = 0.6346$, Pearson $r = 0.0324$ (+0.178 over DeepPrime-base)
- **DeepPrime6-pe6b**: Spearman $\rho = 0.6249$, Pearson $r = 0.0596$ (+0.168 over DeepPrime-base)
- **DeepPrime6-pemaxdrnaseh**: Spearman $\rho = 0.6157$, Pearson $r = 0.0601$ (+0.159 over DeepPrime-base)
- **DeepPrime-base**: Spearman $\rho = 0.4570$, Pearson $r = -0.0039$
- **DeepPrime6-pe6c**: Spearman $\rho = -0.0724$, Pearson $r = -0.2089$
