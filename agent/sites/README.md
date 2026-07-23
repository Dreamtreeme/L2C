# Vision Site Profiles

This directory is the commander-facing site registry for multi-site collection.

The intended flow is:

1. The commander calls a site registry tool to list supported sites.
2. The commander loads one site profile at a time.
3. The site profile is passed to a child vision runner with the user's query.
4. The child runner decides whether to use autonomous exploration or Reflex replay for that site.
5. Collected jobs are persisted in the shared SQLite database.
6. The commander queries the database and summarizes trends for the user.

Files per site:

- `profile.json`: 사이트 정체성, 공식 주소, 화면 역할, 허용 도구, Reflex 경계와 판단 지침의 단일 계약.

`ChatService` is the canonical user-facing orchestrator. `agent.graph.investigation_workflow` resolves material ambiguity, defines evidence requirements, checks DB coverage, and invokes `realtime_scraping` only through a validated action plan. 사이트 선택은 코드가 확정하며 LLM이 스킬 파일을 찾기 위한 별도 호출은 하지 않습니다.

## 공식 시작 주소

각 `profile.json`의 `base_url`은 해당 사이트의 공식 HTTPS 시작 주소입니다. 서버 시작 시 모든 프로필을 Pydantic으로 읽어 URL 패턴, 도구 이름, 중복 도메인을 검증합니다. `get_official_site_url()`은 slug, 한글 이름, 별칭, 도메인 요청을 이 주소로 변환합니다. Vision 작업자는 `open_browser(site=...)`로 이 주소를 먼저 연 뒤 화면 탐색을 시작합니다.
