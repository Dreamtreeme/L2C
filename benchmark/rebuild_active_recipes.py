"""저장된 Critic 판정을 현재 승격 정책으로 다시 적용한다."""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.recipe.candidate_promotion import (
    reapply_reviewed_candidate_promotion,
)
from agent.recipe.candidate_store import RecipeCandidateStore
from agent.recipe.store import RecipeStore


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--site", default="")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    if args.db is None:
        from agent.config import get_settings

        db_path = get_settings().paths.db_path
    else:
        db_path = args.db

    candidates = RecipeCandidateStore(db_path).list_recent(limit=10000, status="accepted")
    if args.site:
        candidates = [item for item in candidates if item.site == args.site]
    candidates.reverse()

    recipe_store = RecipeStore(db_path)
    candidate_sites = {item.site for item in candidates if item.site}
    before = len(recipe_store.get_by_site(args.site)) if args.site else sum(
        len(recipe_store.get_by_site(site)) for site in candidate_sites
    )
    print(f"db={db_path}")
    print(f"site={args.site or '*'} candidates={len(candidates)} active_before={before}")
    if not args.apply:
        print("dry_run=true")
        return 0

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = db_path.with_name(f"{db_path.stem}_before_recipe_rebuild_{stamp}{db_path.suffix}")
    shutil.copy2(db_path, backup_path)
    removed = recipe_store.clear_recipes(args.site or None)
    results = [
        reapply_reviewed_candidate_promotion(item.run_id, db_path=db_path)
        for item in candidates
    ]
    after = len(recipe_store.get_by_site(args.site)) if args.site else sum(
        len(recipe_store.get_by_site(site)) for site in candidate_sites
    )
    print(f"backup={backup_path}")
    print(f"removed={removed} active_after={after}")
    for result in results:
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
