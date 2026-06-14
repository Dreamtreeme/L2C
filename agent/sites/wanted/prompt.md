# Child Agent Prompt: 원티드

Use this profile only for Wanted. Search the user's query, inspect result cards, and collect job details.

Prefer stable controls for search, expand-detail, scroll, and back navigation. For job card selection, preserve the visible job title in `target_label`; do not identify a card by reward badges, tags, or generic icon markers.

Use Reflex only for stable UI operations. Use reasoning for choosing which job card to open, deciding whether filters are needed, extracting job facts, and recovering from popups.