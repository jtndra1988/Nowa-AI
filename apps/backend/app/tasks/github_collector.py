# In app/tasks/github_collector.py
import requests
import json
from datetime import datetime, timezone
from app.celery_app.app import celery_app
from app.db.database import SessionLocal
from app.db.models import DeveloperActivity
from app.exchange.adapters import BinanceDataAdapter, BybitAdapter
from app.core.config import settings

GITHUB_API_BASE_URL = "https://api.github.com/repos/"
REPO_MAPPING_FILE = "/app/app/repo_mapping.json" 

@celery_app.task(name="tasks.collect_github_activity")
def collect_github_activity_task():
    print("[*] Starting GitHub activity collection...")
    db_session = SessionLocal()

    try:
        with open(REPO_MAPPING_FILE, 'r') as f:
            full_repo_mapping = json.load(f)
    except FileNotFoundError:
        print(f"[!] ERROR: repo_mapping.json not found at {REPO_MAPPING_FILE}")
        db_session.close()
        return
    except json.JSONDecodeError:
        print(f"[!] ERROR: Could not decode JSON from repo_mapping.json")
        db_session.close() 
        return

    # ✅ Dynamic Filtering: Only collect GitHub data for active Top 10 symbols
    use_binance = getattr(settings, "USE_BINANCE_FOR_DATA", True)
    if use_binance:
         adapter = BinanceDataAdapter()
    else:
         PAPER_MODE = bool(getattr(settings, "PAPER_TRADING", True))
         adapter = BybitAdapter(paper_mode=PAPER_MODE)

    top_bases = adapter.get_top_symbols_by_volume(limit=10)
    
    # Filter mapping to only include Top 10 active symbols
    target_mapping = {k: v for k, v in full_repo_mapping.items() if k in top_bases}
    
    if not target_mapping:
        print("[i] No matching GitHub repos found for current Top 10 symbols.")
        db_session.close()
        return

    print(f"[*] Collecting GitHub stats for {len(target_mapping)} symbols (Top 10 Filtered)...")

    now_utc = datetime.now(timezone.utc)
    updated_count = 0
    added_count = 0

    try:
        for symbol, repo_url in target_mapping.items():
            repo_path = "/".join(repo_url.split("/")[-2:])
            api_url = f"{GITHUB_API_BASE_URL}{repo_path}"

            try:
                response = requests.get(api_url, timeout=15)
                if response.status_code == 403:
                     print(f"[!] WARN: GitHub rate limit hit for {symbol}. Skipping rest.")
                     break 
                elif response.status_code != 200:
                    print(f"[!] WARN: Failed to fetch {symbol} from GitHub. Status: {response.status_code}")
                    continue
                data = response.json()
            except requests.RequestException as req_e:
                 print(f"[!] WARN: Request failed for {symbol}: {req_e}")
                 continue

            existing_entry = db_session.query(DeveloperActivity)\
                .filter(DeveloperActivity.symbol == symbol)\
                .order_by(DeveloperActivity.timestamp.desc())\
                .first()

            stars_count = data.get('stargazers_count')
            forks_count = data.get('forks_count')
            issues_count = data.get('open_issues_count')

            if existing_entry:
                existing_entry.timestamp = now_utc
                existing_entry.repo_url = repo_url
                existing_entry.stars = stars_count
                existing_entry.forks = forks_count
                existing_entry.open_issues = issues_count
                updated_count += 1
            else:
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
        print(f"[!] ERROR during GitHub activity collection: {e}")
        db_session.rollback()
    finally:
        db_session.close()