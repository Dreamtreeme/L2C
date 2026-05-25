import os
import re
from pathlib import Path
from collections import defaultdict

def clean_json_dir(json_dir: Path):
    print("--- Cleaning up data/json/ ---")
    if not json_dir.exists():
        print("data/json/ does not exist.")
        return

    # Group files by prefix (site_id)
    # File format: <site>_<id>_<timestamp>.json (e.g. rocketpunch_87469_20260514_174643.json)
    groups = defaultdict(list)
    pattern = re.compile(r"^([a-zA-Z0-9]+_\d+|[a-zA-Z0-9]+_unknown)_(\d{8}_\d{6})\.json$")

    for file_path in json_dir.glob("*.json"):
        match = pattern.match(file_path.name)
        if match:
            prefix = match.group(1)
            timestamp = match.group(2)
            groups[prefix].append((timestamp, file_path))
        else:
            # Files that don't match standard pattern
            print(f"Skipping non-standard json file: {file_path.name}")

    for prefix, files in groups.items():
        # Sort by timestamp in descending order (latest first)
        files.sort(key=lambda x: x[0], reverse=True)
        latest_file = files[0][1]
        
        # Keep the latest, delete others
        for timestamp, file_path in files[1:]:
            print(f"Deleting duplicate/older JSON: {file_path.name}")
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path.name}: {e}")

def clean_screenshots_dir(screenshots_dir: Path):
    print("\n--- Cleaning up data/screenshots/ ---")
    if not screenshots_dir.exists():
        print("data/screenshots/ does not exist.")
        return

    # Delete all temporary screen_* and marked_screen_* files
    # Keep only marked_screen.png and marked_test_capture.png
    for file_path in screenshots_dir.glob("*"):
        if file_path.is_file():
            name = file_path.name.lower()
            if name in ("marked_screen.png", "marked_test_capture.png"):
                print(f"Keeping reference screenshot: {file_path.name}")
                continue
            
            if name.startswith("marked_screen_") or name.startswith("screen_"):
                print(f"Deleting temporary/duplicate screenshot: {file_path.name}")
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path.name}: {e}")
            else:
                print(f"Keeping other file: {file_path.name}")

def main():
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    
    clean_json_dir(data_dir / "json")
    clean_screenshots_dir(data_dir / "screenshots")
    print("\nData cleanup completed successfully.")

if __name__ == "__main__":
    main()
