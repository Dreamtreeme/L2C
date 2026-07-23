---
title: "런타임 호환 기준"
type: reference
area: runtime
status: active
updated: 2026-07-23
tags:
  - l2c
  - docs/runtime
---

# 런타임 호환 기준

검증일: 2026-07-23  
대상: Windows 11, NVIDIA GeForce RTX 3080, 드라이버 591.86

## 기준 조합

| 영역 | 버전 | 설치 환경 |
|---|---|---|
| Python | 3.13.14 | 공통 |
| PyTorch / torchvision | 2.13.0+cu130 / 0.28.0+cu130 | `.venv-app` |
| Ultralytics / OpenCV | 8.4.104 / 5.0.0.93 | `.venv-app` |
| LangGraph / SQLite checkpoint / langchain-core | 1.2.9 / 3.1.0 / 1.5.0 | `.venv-app` |
| FastAPI / Pydantic | 0.139.2 / 2.13.4 | `.venv-app` |
| Playwright | 1.61.0 | `.venv-app` |
| PaddlePaddle GPU | 3.3.1, CUDA 13.0, cuDNN 9.13 | `.venv-ocr` |
| PaddleOCR / PaddleX | 3.7.0 / 3.7.2 | `.venv-ocr` |
| OCR용 OpenCV | opencv-contrib-python 4.10.0.84 | `.venv-ocr` |

Python 3.10은 2026년 10월 보안 지원이 끝나므로 새 기준에서 제외했다. Python 3.14 대신 3.13을 선택한 이유는 검증 시점의 PaddlePaddle Windows GPU 안정 휠이 CPython 3.13까지만 제공됐기 때문이다. 기준 Python은 [Python 3.13.14 릴리스](https://www.python.org/downloads/release/python-31314/)이며 지원 기간은 [Python 개발자 가이드](https://devguide.python.org/versions/)에서 확인한다.

## 환경을 나눈 이유

기존 `.venv`에는 PyTorch, PaddlePaddle, 세 종류의 OpenCV 배포판이 함께 설치돼 있었다. 이 때문에 Torch import를 가짜 모듈로 대체하고 CUDA DLL 경로와 import 순서를 조정하는 코드가 필요했다.

새 기준은 다음처럼 소유권을 분리한다.

```text
.venv-app
  PyTorch + Ultralytics + OmniParser + API + LangGraph

.venv-ocr
  PaddlePaddle + PaddleOCR + PaddleX
```

OpenCV 공식 Python 패키지는 `cv2` 네임스페이스를 공유하는 배포판을 한 환경에 둘 이상 설치하지 말라고 명시한다. 따라서 앱에는 `opencv-python`, OCR 환경에는 PaddleX가 요구하는 `opencv-contrib-python`만 둔다. 근거는 [OpenCV Python 패키지 안내](https://pypi.org/project/opencv-python/)를 따른다.

## PaddleOCR 전환

PaddleOCR 3.x는 2.x와 호환되지 않는다. 기존 `PaddleOCR(... use_gpu=...)`와 `ocr.ocr(..., cls=False)` 호출을 제거하고 다음 3.x 계약으로 바꿨다.

- 한국어 인식은 `lang="korean"`, `ocr_version="PP-OCRv5"`를 사용한다.
- 추론은 `predict()`로 실행한다.
- 결과 객체의 `rec_texts`, `rec_scores`, `rec_boxes`를 기존 마커 형식으로 변환한다.
- 문서 방향 분류, 문서 왜곡 보정, 텍스트 줄 방향 분류는 브라우저 화면 OCR에 필요하지 않아 끈다.

공식 근거는 [PaddleOCR 3.x 설치](https://www.paddleocr.ai/main/en/version3.x/installation.html), [OCR 파이프라인 사용법](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/OCR.html), [2.x에서 3.x 업그레이드 안내](https://www.paddleocr.ai/main/en/update/upgrade_notes.html)를 따른다.

## CUDA 조합 검증

| 후보 | 결과 | 판정 |
|---|---|---|
| Paddle 3.2.0 cu118 | 철회된 cuDNN 8.9 패키지 의존성 경고 | 제외 |
| Paddle 3.3.1 cu126 | OCR은 성공했지만 컴파일 cuDNN 9.9와 설치 cuDNN 9.5 불일치 경고 | 제외 |
| Paddle 3.3.1 cu130 | CUDA 13.0, cuDNN 9.13 일치. DLL 디렉터리 등록 후 오류·불일치 경고 없음 | 채택 |

Paddle 공식 저장소의 CUDA 13.0 Windows CPython 3.13 휠을 사용한다. 설치 가능 파일은 [Paddle cu130 인덱스](https://www.paddlepaddle.org.cn/packages/stable/cu130/paddlepaddle-gpu/)에서 확인한다. Windows 휠의 NVIDIA DLL은 `.venv-ocr/Lib/site-packages/nvidia` 아래에 있으므로 OCR 작업자가 Paddle import 전에 해당 디렉터리를 프로세스 DLL 검색 경로에 등록한다.

## 검증 결과

- `.venv-app`, `.venv-ocr` 모두 `pip check` 통과
- PyTorch CUDA 텐서 연산 통과
- 기존 OmniParser `model.pt` GPU 추론 통과, 테스트 화면에서 63개 상자 검출
- Paddle CUDA 텐서 연산 통과
- 사람인 한국어 화면에서 OCR 113개 상자 반환
- 동일 OCR 작업자 재사용: 첫 요청 1.63초, 두 번째 요청 0.59초, 작업자 세대 1 유지
- Python 3.13 전체 에이전트 테스트: 380개 통과

LangGraph 설치 기준은 [공식 설치 문서](https://docs.langchain.com/oss/python/langgraph/install), 영속 체크포인트는 [공식 SQLite 체크포인터](https://pypi.org/project/langgraph-checkpoint-sqlite/), PyTorch CUDA 휠은 [공식 cu130 인덱스](https://download.pytorch.org/whl/cu130/), Playwright 브라우저 설치는 [공식 Python 문서](https://playwright.dev/python/docs/intro)를 기준으로 했다. 최신 Starlette 테스트 클라이언트는 [공식 문서](https://www.starlette.io/testclient/)에 따라 개발 환경에서 `httpx2`를 사용한다.

## 설치와 검사

```powershell
.\setup.cmd
.\.venv-app\Scripts\python.exe scripts\check_runtime_compat.py --profile app
.\.venv-ocr\Scripts\python.exe scripts\check_runtime_compat.py --profile ocr
.\scripts\test.cmd agent\tests -q
```

`setup.cmd`는 공식 SHA-256을 검증한 Python 3.13.14 설치, 앱·OCR 환경 구성, Chromium과 모델 다운로드, GPU 호환성 검사를 순서대로 실행한다. NVIDIA 드라이버만 자동 설치 대상에서 제외한다. 설치 전 동작만 확인하려면 `setup.cmd -DryRun`을 사용한다.

`requirements.txt`는 앱, `requirements-ocr.txt`는 독립 OCR 작업자, `requirements-dev.txt`는 테스트와 모델 벤치마크 의존성을 소유한다.
