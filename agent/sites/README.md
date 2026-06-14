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
- `prompt.md`: short child-agent instruction block for that site.
- `tools.json`: allowed tool policy and Reflex boundaries.

`realtime_scraping` can load these profiles to build site-specific collection goals. `agent.graph.commander_workflow` now provides the top-level LangGraph fan-out path: plan sites, run a child worker, review the structured submission, retry on feedback, persist accepted data, then query DB evidence for summarization.