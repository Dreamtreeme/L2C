# E2E 실행 기록 감사

- 생성 시각: `2026-07-28T18:00:09.438186+09:00`
- 탐색한 summary: `143`개
- 수집 E2E 기록: `92`개
- 기준 커밋 후보(clean): `14`개
- 개발 중 기록(dirty): `78`개
- 식별자 불완전: `0`개
- 비교 가능한 반복 그룹: `17`개
- 수집 E2E가 아닌 summary: `51`개

## 해석 기준

- `release`: 깨끗한 작업 트리에서 실행되어 최종 기준값 후보로 사용할 수 있다.
- `development`: 변경 파일이 있는 상태의 실행이다. 성능 기준값이 아니라 회귀·트러블슈팅 증거로 사용한다.
- 같은 분류, 커밋, 설정 fingerprint, 시나리오, 사이트, 실행 모드, 질의, 목표 수만 한 그룹으로 묶는다.
- 작은 표본에는 p95를 계산하지 않고 성공 건수와 실행시간 최소·중앙·최대값을 그대로 표시한다.

## 비교 그룹

| 분류 | 커밋 | 설정 | 시나리오 | 모드 | 성공/전체 | 실행시간 최소/중앙/최대(초) | 토큰 중앙 | 비용 중앙($) | Reflex 중앙 |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| release | `005e0e15` | `62880d31ed47` | jobkorea_backend1_generality | cold | 1/1 | 48.73/48.73/48.73 | 50861 | 0.0757 | 0.0 |
| release | `dc4eda11` | `63e95ccaade5` | saramin-ai-cold | cold | 1/1 | 38.72/38.72/38.72 | 44766 | 0.0653 | 0.0 |
| release | `dc4eda11` | `68e2c6611000` | saramin-ai-warm | warm | 1/1 | 32.11/32.11/32.11 | 50644 | 0.0757 | 0.0 |
| release | `7408d08e` | `62880d31ed47` | saramin-ml1-generality | cold | 1/1 | 50.71/50.71/50.71 | 52545 | 0.0761 | 0.0 |
| release | `014a5f08` | `a9448dbeb474` | atomic-wanted-ios2-warm-retry | warm | 0/1 | 110.18/110.18/110.18 | 145345 | 0.2305 | 0.0 |
| release | `5ce8d98c` | `68e2c6611000` | capture-id-wanted-ios1 | warm | 1/1 | 29.27/29.27/29.27 | 32608 | - | 0.0 |
| release | `88a5288e` | `62880d31ed47` | capturefix-wanted-ios2-cold | cold | 1/1 | 78.96/78.96/78.96 | 98639 | 0.1473 | 0.0 |
| release | `7408d08e` | `a9448dbeb474` | wanted-data-engineer2-parameterized | warm | 1/1 | 79.35/79.35/79.35 | 94280 | 0.1395 | 3.0 |
| release | `dc4eda11` | `63e95ccaade5` | wanted-ios-cold | cold | 1/1 | 60.62/60.62/60.62 | 86169 | 0.1274 | 0.0 |
| release | `dc4eda11` | `68e2c6611000` | wanted-ios-warm | warm | 1/1 | 53.39/53.39/53.39 | 74766 | 0.1129 | 0.0 |
| release | `7408d08e` | `a9448dbeb474` | wanted-ios1-count | warm | 1/1 | 43.47/43.47/43.47 | 33008 | 0.0459 | 4.0 |
| release | `005e0e15` | `62880d31ed47` | work24_data1_generality | cold | 1/1 | 83.12/83.12/83.12 | 125499 | 0.1924 | 0.0 |
| release | `dc4eda11` | `63e95ccaade5` | worknet-data-cold | cold | 1/1 | 61.36/61.36/61.36 | 102935 | 0.1572 | 0.0 |
| release | `dc4eda11` | `68e2c6611000` | worknet-data-warm | warm | 1/1 | 54.36/54.36/54.36 | 94250 | 0.1425 | 1.0 |
| development | `9e2fce9b` | `63e95ccaade5` | wanted-ios-cold | cold | 4/7 | 8.15/65.02/95.99 | 80068 | - | 0.0 |
| development | `9e2fce9b` | `68e2c6611000` | wanted-ios-warm | warm | 3/5 | 7.86/42.04/64.35 | 50638 | - | 0.0 |
| development | `9e2fce9b` | `68e2c6611000` | worknet-data-warm | warm | 1/4 | 7.18/58.82/134.03 | 56886 | - | 0.0 |
| development | `9e2fce9b` | `63e95ccaade5` | saramin-ai-cold | cold | 2/3 | 7.39/36.66/45.53 | 45090 | - | 0.0 |
| development | `9e2fce9b` | `68e2c6611000` | saramin-ai-warm | warm | 2/3 | 7.40/34.03/48.11 | 35551 | - | 0.0 |
| development | `9e2fce9b` | `63e95ccaade5` | worknet-data-cold | cold | 1/3 | 7.18/46.49/137.42 | 64473 | - | 0.0 |
| development | `c6633b7d` | `a9448dbeb474` | jobkorea-backend-warm | warm | 1/2 | 22.47/28.49/34.52 | 22688 | 0.0335 | 1.0 |
| development | `4bb93faa` | `cb37aefb6d09` | jobkorea-data-engineer-2 | warm | 1/2 | 77.34/113.94/150.53 | 85101 | - | 1.0 |
| development | `4bb93faa` | `cb37aefb6d09` | jobkorea-ios-2 | cold | 1/2 | 103.22/164.98/226.75 | 138916 | - | 0.0 |
| development | `4bb93faa` | `cb37aefb6d09` | jobkorea-ml-engineer-1 | warm | 2/2 | 32.95/38.76/44.57 | 11468 | - | 1.0 |
| development | `9e2fce9b` | `63e95ccaade5` | rocketpunch-backend-cold | cold | 1/2 | 7.39/20.05/32.71 | 19016 | - | 0.0 |
| development | `9e2fce9b` | `68e2c6611000` | rocketpunch-backend-warm | warm | 1/2 | 7.37/18.10/28.84 | 9806 | - | 0.5 |
| development | `c6633b7d` | `62880d31ed47` | saramin-ai-cold | cold | 2/2 | 55.24/59.22/63.20 | 64796 | 0.0932 | 0.0 |
| development | `c6633b7d` | `a9448dbeb474` | saramin-ai-warm | warm | 2/2 | 47.51/65.09/82.67 | 54990 | 0.0802 | 3.0 |
| development | `c6633b7d` | `62880d31ed47` | wanted-ios-cold | cold | 2/2 | 79.84/80.81/81.78 | 98848 | 0.1469 | 0.0 |
| development | `ce9a1cfa` | `a9448dbeb474` | wanted-ios-warm | warm | 2/2 | 49.70/51.08/52.45 | 64743 | 0.0926 | 2.5 |
| development | `c6633b7d` | `a9448dbeb474` | wanted-ios-warm | warm | 2/2 | 72.27/73.57/74.87 | 73814 | 0.1088 | 4.0 |
| development | `9e2fce9b` | `6860cb5fd0f9` | duplicate-stop-jobkorea-ml1 | cold | 1/1 | 29.21/29.21/29.21 | 30552 | - | 0.0 |
| development | `c6633b7d` | `62880d31ed47` | jobkorea-backend-cold | cold | 1/1 | 76.12/76.12/76.12 | 99580 | 0.1533 | 0.0 |
| development | `9e2fce9b` | `6860cb5fd0f9` | rocketpunch-backend1-cold | cold | 1/1 | 65.30/65.30/65.30 | 70723 | - | 0.0 |
| development | `9e2fce9b` | `6860cb5fd0f9` | rocketpunch-backend1-warm | warm | 1/1 | 33.64/33.64/33.64 | 50319 | - | 0.0 |
| development | `9e2fce9b` | `6860cb5fd0f9` | card-identity-stop-saramin-ml1 | cold | 1/1 | 37.40/37.40/37.40 | 49693 | - | 0.0 |
| development | `9e2fce9b` | `6860cb5fd0f9` | card-identity-stop-saramin-ml1-verified | cold | 1/1 | 41.66/41.66/41.66 | 53563 | - | 0.0 |
| development | `9e2fce9b` | `6860cb5fd0f9` | duplicate-stop-saramin-ml1 | cold | 1/1 | 38.58/38.58/38.58 | 51378 | - | 0.0 |
| development | `ce9a1cfa` | `a9448dbeb474` | saramin-ai-warm | warm | 1/1 | 31.94/31.94/31.94 | 35368 | 0.0498 | 1.0 |
| development | `828cf448` | `cb37aefb6d09` | saramin-ml-engineer-1 | cold | 1/1 | 107.55/107.55/107.55 | 97477 | - | 0.0 |
| development | `828cf448` | `cb37aefb6d09` | saramin-ml-engineer-1-startup-fix | cold | 1/1 | 165.91/165.91/165.91 | 62384 | - | 0.0 |
| development | `9e2fce9b` | `6860cb5fd0f9` | saramin-solbit-ml1-warm | warm | 1/1 | 25.36/25.36/25.36 | 30789 | - | 0.0 |
| development | `7408d08e` | `62880d31ed47` | saramin_ml1_urlsync | cold | 1/1 | 54.02/54.02/54.02 | 55967 | 0.0769 | 0.0 |
| development | `9e2fce9b` | `68e2c6611000` | capture-id-wanted-ios1 | warm | 0/1 | 49.27/49.27/49.27 | 0 | - | 0.0 |
| development | `9e2fce9b` | `10315e64ff6f` | duplicate-stop-low-thinking | cold | 0/1 | 29.20/29.20/29.20 | 14702 | - | 4.0 |
| development | `9e2fce9b` | `6860cb5fd0f9` | duplicate-stop-low-thinking-verified | cold | 1/1 | 51.67/51.67/51.67 | 14703 | - | 4.0 |
| development | `9e2fce9b` | `10315e64ff6f` | lite-stream-cap-1 | cold | 0/1 | 59.32/59.32/59.32 | 37935 | - | 4.0 |
| development | `9e2fce9b` | `10315e64ff6f` | lite-stream-cap-detail-1 | cold | 1/1 | 60.46/60.46/60.46 | 61406 | - | 4.0 |
| development | `9e2fce9b` | `10315e64ff6f` | model-swap-new-1 | cold | 0/1 | 73.88/73.88/73.88 | 37890 | - | 4.0 |
| development | `ce9a1cfa` | `62880d31ed47` | wanted-ios-cold | cold | 1/1 | 62.18/62.18/62.18 | 86169 | 0.1274 | 0.0 |
| development | `88a5288e` | `a9448dbeb474` | wanted-ios2-critic-contract | warm | 1/1 | 76.13/76.13/76.13 | 73975 | 0.1071 | 5.0 |
| development | `88a5288e` | `a9448dbeb474` | wanted-ios2-recipe-dedupe | warm | 1/1 | 65.60/65.60/65.60 | 63711 | 0.0937 | 4.0 |
| development | `7f945327` | `62880d31ed47` | wanted-ios2-roi3-cold | cold | 0/1 | 291.40/291.40/291.40 | 540849 | 0.8289 | 0.0 |
| development | `7f945327` | `62880d31ed47` | wanted-ios2-roi3-cold-retry | cold | 1/1 | 58.88/58.88/58.88 | 93783 | 0.1393 | 0.0 |
| development | `7f945327` | `a9448dbeb474` | wanted-ios2-roi3-warm | warm | 1/1 | 58.26/58.26/58.26 | 77800 | 0.1129 | 3.0 |
| development | `7f945327` | `68e2c6611000` | wanted-ios2-roi3-warm-clean | warm | 1/1 | 51.91/51.91/51.91 | 64653 | 0.0957 | 3.0 |
| development | `7f945327` | `68e2c6611000` | wanted-ios2-roi3-warm-contract | warm | 1/1 | 50.36/50.36/50.36 | 64746 | 0.0964 | 3.0 |
| development | `9e2fce9b` | `6860cb5fd0f9` | duplicate-stop-work24-erp1 | unspecified | 0/1 | 104.21/104.21/104.21 | 232414 | - | 0.0 |
| development | `9e2fce9b` | `6860cb5fd0f9` | duplicate-stop-work24-erp1-fixed | unspecified | 1/1 | 74.79/74.79/74.79 | 126786 | - | 0.0 |
| development | `005e0e15` | `62880d31ed47` | work24_data1_popupfix | cold | 1/1 | 47.16/47.16/47.16 | 50221 | 0.0740 | 0.0 |
| development | `c6633b7d` | `62880d31ed47` | worknet-data-cold | cold | 1/1 | 59.43/59.43/59.43 | 70473 | 0.1051 | 0.0 |
| development | `c6633b7d` | `a9448dbeb474` | worknet-data-warm | warm | 1/1 | 64.04/64.04/64.04 | 70970 | 0.1053 | 1.0 |
