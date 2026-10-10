# Orthogonal Benchmark Predictions & Visualizations

외부 공식 벤치마크 테스트셋(DeepPrime Generalization Test Set, PRIDICT2.0 23k Fold 0 Test Set)에 대한 모델 예측값, 상관계수 히트맵 및 최적화 시각화 결과물입니다.

---

## 1. 생성된 시각화 플롯 목록 (300 DPI)

### A. 상관계수 히트맵 (Ground Truth Correlation & Pairwise Matrices) - 권장
- [`ground_truth_correlation_heatmap.png`](file:///home/work/workdir/deepprime6-genomebiol-revision/notebooks/plots/orthogonal_benchmark/ground_truth_correlation_heatmap.png):
  **[핵심 종합도]** 평가 모델 6종(행)과 두 벤치마크의 Spearman ($\rho$) / Pearson ($r$)(열)에 대한 Ground Truth 상관계수 히트맵
- [`spearman_heatmap_pairwise_matrices.png`](file:///home/work/workdir/deepprime6-genomebiol-revision/notebooks/plots/orthogonal_benchmark/spearman_heatmap_pairwise_matrices.png):
  두 벤치마크 각각에서의 실측값 및 모델 6종 간 $7 \times 7$ pairwise Spearman correlation matrix 히트맵 (2-패널)
- [`pearson_heatmap_pairwise_matrices.png`](file:///home/work/workdir/deepprime6-genomebiol-revision/notebooks/plots/orthogonal_benchmark/pearson_heatmap_pairwise_matrices.png):
  두 벤치마크 각각에서의 실측값 및 모델 6종 간 $7 \times 7$ pairwise Pearson correlation matrix 히트맵 (2-패널)

### B. 10분위수 캘리브레이션 곡선 (Decile Calibration Curves) - Boxplot 스케일 한계 극복
- [`decile_calibration_curves.png`](file:///home/work/workdir/deepprime6-genomebiol-revision/notebooks/plots/orthogonal_benchmark/decile_calibration_curves.png):
  모델별 예측값을 10분위수(Decile, 1~10)로 순위화한 후, 각 분위수별 실제 측정 효율 평균을 플롯.
  *단순 수치 범위 차이(0~100% vs 0~5%)로 인한 boxplot 왜곡을 완전히 해결하고 모델의 실질적 순위 판별력(monotonic ranking power)을 명확히 제시.*

### C. 모델별 성능 비교 막대 그래프 & Boxplots
- [`barplot_correlation_comparison.png`](file:///home/work/workdir/deepprime6-genomebiol-revision/notebooks/plots/orthogonal_benchmark/barplot_correlation_comparison.png):
  두 벤치마크에 대한 Spearman $\rho$ 및 Pearson $r$ 상관계수 수치 비교 막대 그래프
- [`boxplot_combined_benchmarks.png`](file:///home/work/workdir/deepprime6-genomebiol-revision/notebooks/plots/orthogonal_benchmark/boxplot_combined_benchmarks.png):
  두 벤치마크 2-패널 통합 예측 분포 Boxplot
- [`boxplot_deepprime_test_predictions.png`](file:///home/work/workdir/deepprime6-genomebiol-revision/notebooks/plots/orthogonal_benchmark/boxplot_deepprime_test_predictions.png)
- [`boxplot_pridict2_test_predictions.png`](file:///home/work/workdir/deepprime6-genomebiol-revision/notebooks/plots/orthogonal_benchmark/boxplot_pridict2_test_predictions.png)

---

## 2. 수치 데이터 및 상관계수 행렬 CSV 파일
- [`ground_truth_correlations_summary.csv`](file:///home/work/workdir/deepprime6-genomebiol-revision/notebooks/plots/orthogonal_benchmark/ground_truth_correlations_summary.csv): 실측값 대비 모델별 상관계수 요약표
- [`correlation_matrix_deepprime_spearman.csv`](file:///home/work/workdir/deepprime6-genomebiol-revision/notebooks/plots/orthogonal_benchmark/correlation_matrix_deepprime_spearman.csv): DeepPrime 테스트셋 pairwise Spearman 행렬
- [`correlation_matrix_deepprime_pearson.csv`](file:///home/work/workdir/deepprime6-genomebiol-revision/notebooks/plots/orthogonal_benchmark/correlation_matrix_deepprime_pearson.csv): DeepPrime 테스트셋 pairwise Pearson 행렬
- [`correlation_matrix_pridict2_spearman.csv`](file:///home/work/workdir/deepprime6-genomebiol-revision/notebooks/plots/orthogonal_benchmark/correlation_matrix_pridict2_spearman.csv): PRIDICT2.0 테스트셋 pairwise Spearman 행렬
- [`correlation_matrix_pridict2_pearson.csv`](file:///home/work/workdir/deepprime6-genomebiol-revision/notebooks/plots/orthogonal_benchmark/correlation_matrix_pridict2_pearson.csv): PRIDICT2.0 테스트셋 pairwise Pearson 행렬
- [`deepprime_test_raw_predictions.csv`](file:///home/work/workdir/deepprime6-genomebiol-revision/notebooks/plots/orthogonal_benchmark/deepprime_test_raw_predictions.csv): DeepPrime 테스트셋($N = 7,243$) 원본 예측치
- [`pridict2_test_raw_predictions.csv`](file:///home/work/workdir/deepprime6-genomebiol-revision/notebooks/plots/orthogonal_benchmark/pridict2_test_raw_predictions.csv): PRIDICT2.0 테스트셋($N = 4,311$) 원본 예측치
- [`performance_summary.csv`](file:///home/work/workdir/deepprime6-genomebiol-revision/notebooks/plots/orthogonal_benchmark/performance_summary.csv): 최종 모델 성능 요약표
- [`overlap_exclusion_summary.csv`](file:///home/work/workdir/deepprime6-genomebiol-revision/notebooks/plots/orthogonal_benchmark/overlap_exclusion_summary.csv): Stratification 필터링 집계표
