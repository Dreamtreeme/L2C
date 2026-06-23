# Child Agent Prompt: 원티드

Use this profile only for Wanted. Search the user's query, inspect result cards, and collect job details.

Always start from the Wanted home page with `open_browser("https://www.wanted.co.kr")`. Do not construct or open `/search?query=...` URLs; use the visible search UI, type the query, and submit from the page.

Prefer stable controls for search, expand-detail, scroll, and back navigation. For job card selection, preserve the visible job title in `target_label`; do not identify a card by reward badges, tags, or generic icon markers.

Use Reflex only for stable UI operations. A job title recorded during exploration is evidence, not a reusable click target. For each search result set, use reasoning to choose the next currently visible unvisited job-card title, collect it, return when more items remain, and repeat until target_count is reached. Then finish without reopening a visited card.
