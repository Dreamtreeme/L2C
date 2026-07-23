# L2C 프론트엔드

React와 TypeScript로 구현한 로컬 조사 작업공간입니다. FastAPI의 기존 채팅 SSE, 확인 질문, 취소, 공고 출처 및 운영 API를 그대로 사용합니다.

## 개발

백엔드를 `127.0.0.1:8000`에서 실행한 뒤 다음 명령을 사용합니다.

```powershell
cd frontend
npm ci
npm run dev
```

Vite 개발 서버는 `/api` 요청을 로컬 FastAPI로 전달합니다.

백엔드나 외부 모델 호출 없이 전체 UI 상태만 확인할 때는 별도 터미널에서 다음 모의 API를 실행합니다.

```powershell
npm run mock-api
```

## 검증

```powershell
npm test
npm run build
```

저장소 루트의 `run.cmd`는 필요한 경우 프론트 빌드를 갱신하고 FastAPI를 시작한 뒤 브라우저를 엽니다.
