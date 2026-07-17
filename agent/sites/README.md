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

- `manual.json`: structured site strategy and known stable/variable UI concepts.
- `SKILL.md`: 선택된 사이트 작업자에게만 주입하는 짧은 판단 지침.
- `tools.json`: allowed tool policy and Reflex boundaries.

`ChatService` is the canonical user-facing orchestrator. `agent.graph.investigation_workflow` resolves material ambiguity, defines evidence requirements, checks DB coverage, and invokes `realtime_scraping` only through a validated action plan. 사이트 선택은 코드가 확정하며 LLM이 스킬 파일을 찾기 위한 별도 호출은 하지 않습니다.

## 공식 시작 주소

`registry.json`의 `base_url`은 각 사이트의 공식 HTTPS 시작 주소를 나타내는 단일 기준입니다. `get_official_site_url()`은 slug, 한글 이름, 별칭, 도메인 요청을 이 주소로 변환하고 등록 도메인과 일치하는지 검증합니다. Vision 작업자는 LLM에게 주소 선택을 맡기지 않고 `open_browser(site=...)`로 이 주소를 먼저 연 뒤 화면 탐색을 시작합니다.
