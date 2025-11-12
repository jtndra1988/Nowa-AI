# In app/tasks/github_collector.py
import requests
import json
from datetime import datetime, timezone # Ensure timezone is imported
from app.celery_app.app import celery_app
from app.db.database import SessionLocal # Use SessionLocal consistently
from app.db.models import DeveloperActivity

GITHUB_API_BASE_URL = "https://api.github.com/repos/"
REPO_MAPPING_FILE = "/app/app/repo_mapping.json" # Ensure this path is correct inside Docker

@celery_app.task(name="tasks.collect_github_activity")
def collect_github_activity_task():
    print("[*] Starting GitHub activity collection...")
    db_session = SessionLocal()

    try:
        with open(REPO_MAPPING_FILE, 'r') as f:
            repo_mapping = json.load(f)
    except FileNotFoundError:
        print(f"[!] ERROR: repo_mapping.json not found at {REPO_MAPPING_FILE}")
        db_session.close() # Close session even on error
        return
    except json.JSONDecodeError:
        print(f"[!] ERROR: Could not decode JSON from repo_mapping.json")
        db_session.close() # Close session even on error
        return

    now_utc = datetime.now(timezone.utc) # Use timezone-aware timestamp
    updated_count = 0
    added_count = 0

    try:
        for symbol, repo_url in repo_mapping.items():
            repo_path = "/".join(repo_url.split("/")[-2:])
            api_url = f"{GITHUB_API_BASE_URL}{repo_path}"

            try:
                response = requests.get(api_url, timeout=15) # Add timeout
                # Handle rate limits or errors
                if response.status_code == 403:
                     print(f"[!] WARN: GitHub rate limit hit for {symbol}. Skipping rest.")
                     break # Stop processing if rate limited
                elif response.status_code != 200:
                    print(f"[!] WARN: Failed to fetch data for {symbol} from {api_url}. Status: {response.status_code}")
                    continue
                data = response.json()
            except requests.RequestException as req_e:
                 print(f"[!] WARN: Request failed for {symbol} ({api_url}): {req_e}")
                 continue

            # Find existing record (simplest: find latest for the symbol)
            existing_entry = db_session.query(DeveloperActivity)\
                .filter(DeveloperActivity.symbol == symbol)\
                .order_by(DeveloperActivity.timestamp.desc())\
                .first()

            stars_count = data.get('stargazers_count')
            forks_count = data.get('forks_count')
            issues_count = data.get('open_issues_count')

            # Create or Update
            if existing_entry:
                # Update existing record's stats and timestamp
                existing_entry.timestamp = now_utc
                existing_entry.repo_url = repo_url # Update in case it changed
                existing_entry.stars = stars_count
                existing_entry.forks = forks_count
                existing_entry.open_issues = issues_count
                updated_count += 1
            else:
                # Add new record
                new_entry = DeveloperActivity(
                    symbol=symbol,
                    timestamp=now_utc,
                    repo_url=repo_url,
                    stars=stars_count,
                    forks=forks_count,
                    open_issues=issues_count
                )
                db_session.add(new_entry)
                added_count += 1

        db_session.commit()
        print(f"[✔] GitHub activity: Added {added_count}, Updated {updated_count} symbols.")

    except Exception as e:
        print(f"[!] ERROR during GitHub activity collection/commit: {e}")
        db_session.rollback()
    finally:
        db_session.close()