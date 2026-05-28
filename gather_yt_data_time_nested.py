# import requests, yaml
import argparse
import pandas as pd
import time
from tqdm import tqdm
import googleapiclient.discovery
import subprocess

import time
import os
from dotenv import load_dotenv
import re
import json
import time
import pandas as pd
from datetime import datetime, timezone, timedelta

from yt_dlp import YoutubeDL
from youtube_comment_downloader import YoutubeCommentDownloader
from youtube_comment_downloader import SORT_BY_POPULAR

parser = argparse.ArgumentParser()
parser.add_argument('--newspaper_source', default="all", help="Which newspaper(s) to process, comma-separated (e.g. 'repubblica,il_giornale') or 'all'")
parser.add_argument('--api_key_name', default="youtube_api", help="Name of the environment variable containing the YouTube API key (default: youtube_api)")
args = parser.parse_args()

GENERAL_CHANNEL_URL = "https://www.youtube.com/@{CHANNEL_NAME}"

newspapers = {
    "repubblica": "@repubblica",
    "corriere_della_sera": "@CorrieredellaSera",
    # "la_gazzetta_dello_sport": "@LaGazzettadelloSport", Removed, shorts contain commentaries
    # "il_sole_24_ore": "@ilsole24ore",  Removed, shorts contain commentaries
    # "avvenireNEI": "@AvvenireNEI", Removed, shorts contain commentaries
    "lastampa": "@LaStampa",
    # "il_fatto_quotidiano": "@IlFattoQuotidiano", # Removed, shorts contain commentaries
    "ilmessaggero": "@ilmessaggero",
    # "il_quotidiano_nazionale": "@quotidianonet", # Removed, shorts contain commentaries
    "il_gazzettino": "@ilgazzettino",
    # "il_giornale": "@ilgiornale" # Removed, shorts contain commentaries
}

# Replace with your own API Key
load_dotenv()
DEVELOPER_KEY = os.getenv(args.api_key_name)

# Initialize YouTube API client
youtube = googleapiclient.discovery.build(
    "youtube", "v3", developerKey=DEVELOPER_KEY)

def build_video_info(item, newspaper):
    """Build the video_info dict from an already-fetched API item."""
    snippet = item["snippet"]
    stats = item["statistics"]
    return {
        'video_id':      item['id'],
        'newspaper':     newspaper,
        'title':         snippet['title'],
        'channel':       snippet['channelTitle'],
        'published_at':  snippet['publishedAt'],
        'description':   snippet.get('description', ''),
        'tags':          ', '.join(snippet.get('tags', [])),
        'view_count':    int(stats.get('viewCount', 0)),
        'like_count':    int(stats.get('likeCount', 0)),
        'comment_count': int(stats.get('commentCount', 0)),
    }


def get_shorts_from_channel(channel_url, newspaper, max_shorts=100, min_age_days=1):
    """
    Extract Shorts from a channel, keeping only those at least `min_age_days` old. 
    """
    channel_url = GENERAL_CHANNEL_URL.format(CHANNEL_NAME=channel_url.lstrip("@"))
    shorts_url = channel_url.rstrip("/") + "/shorts"

    ydl_opts = {"extract_flat": True, "quiet": True, "skip_download": True}
    cutoff = datetime.now(timezone.utc) - timedelta(days=min_age_days)
    shorts = []
    candidate_ids = []

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(shorts_url, download=False)
        print(f"Info extracted for {shorts_url}")
        if "entries" not in info:
            print("No shorts found.")
            return shorts
        for entry in info["entries"]:
            if entry and entry.get("id"):
                candidate_ids.append(entry["id"])

    # Batch the API calls: videos().list accepts up to 50 ids per request
    for i in range(0, len(candidate_ids), 50):
        if len(shorts) >= max_shorts:
            break
        batch = candidate_ids[i:i + 50]
        try:
            response = youtube.videos().list(
                part="snippet,statistics", id=",".join(batch)
            ).execute()
        except Exception as e:
            print(f"  Skipping batch {i}: {e}")
            continue

        for item in response["items"]:
            if len(shorts) >= max_shorts:
                break
            published_at = item["snippet"]["publishedAt"]
            pub_dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
            if pub_dt > cutoff:
                continue  # too recent
            shorts.append(build_video_info(item, newspaper))
            if len(shorts) % 10 == 0:
                print(f"Found {len(shorts)} eligible shorts so far...")

    if len(shorts) < max_shorts:
        print(f"  Warning: only {len(shorts)} shorts older than {min_age_days} "
              f"day(s) (wanted {max_shorts}).")
    return shorts


def _norm_handle(s: str) -> str:
    return str(s).lstrip('@').rstrip(',.:;!?').lower()


def _scrape_replies_yt_dlp(video_id, max_total="all", max_parents="all", max_replies="all"):
    """yt-dlp web-scrape to recover deep replies the API can't return."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    mc = f"{max_total},{max_parents},{max_replies},{max_replies}"
    cmd = [
        "yt-dlp", "--write-comments", "--skip-download",
        "--extractor-args", f"youtube:max_comments={mc};comment_sort=top",
        "--dump-single-json", url,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            print(f"  yt-dlp error for {video_id}: {r.stderr.strip()[:200]}")
            return []
        return json.loads(r.stdout).get("comments", [])
    except Exception as e:
        print(f"  yt-dlp exception for {video_id}: {e}")
        return []


def _infer_threading(df):
    """Recover reply-to-reply nesting from @mentions + timestamps. Adds inferred_parent_id, depth."""
    if df.empty:
        df["inferred_parent_id"] = []
        df["depth"] = []
        return df

    df = df.copy()
    df["_ts"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    df = df.sort_values("_ts").reset_index(drop=True)
    df["author_handle"] = df["author"].map(_norm_handle)
    df["inferred_parent_id"] = df["parent_comment_id"]

    for thread_id, group in df.groupby("parent_comment_id", dropna=False):
        thread_rows = group.sort_values("_ts")
        known = set(thread_rows["author_handle"])
        for idx in group[group["is_reply"]].index:
            row = df.loc[idx]
            text = str(row["text"]).replace("\xa0", " ").replace("\u200b", "").strip()
            m = re.match(r"\s*@([^\s,.:;!?]+)", text)
            if not m:
                continue
            token = _norm_handle(m.group(1))
            matches = [h for h in known if token.startswith(h)]
            if not matches:
                continue
            target = max(matches, key=len)
            cand = thread_rows[
                (thread_rows["author_handle"] == target)
                & (thread_rows["_ts"] < row["_ts"])
                & (thread_rows.index != idx)
            ]
            if not cand.empty:
                df.at[idx, "inferred_parent_id"] = cand.iloc[-1]["comment_id"]

    id_to_parent = dict(zip(df["comment_id"], df["inferred_parent_id"]))
    top_ids = set(df.loc[~df["is_reply"], "comment_id"])

    def depth(cid):
        d, seen = 0, set()
        while cid in id_to_parent and cid not in top_ids and cid not in seen:
            seen.add(cid)
            cid = id_to_parent[cid]
            d += 1
        return d

    df["depth"] = df["comment_id"].map(depth)
    return df.drop(columns=["_ts", "author_handle"])

def fetch_comments(video_id, max_comments):
    """API pass (top-level + bundled replies) merged with yt-dlp deep replies,
    then nested threading inferred from @mentions. Returns list of dicts."""
    api_rows = []
    next_page_token = None

    # 1) API pass
    while len(api_rows) < max_comments:
        response = youtube.commentThreads().list(
            part="snippet,replies",
            videoId=video_id,
            maxResults=100,
            pageToken=next_page_token,
            order="time",
        ).execute()

        for item in response["items"]:
            top = item["snippet"]["topLevelComment"]
            tid = top["id"]
            ts = top["snippet"]
            api_rows.append({
                "comment_id": tid, "parent_comment_id": None, "video_id": video_id,
                "is_reply": False, "author": ts["authorDisplayName"],
                "published_at": ts["publishedAt"], "updated_at": ts["updatedAt"],
                "like_count": ts["likeCount"], "text": ts["textOriginal"],
            })
            for reply in item.get("replies", {}).get("comments", []):
                rs = reply["snippet"]
                api_rows.append({
                    "comment_id": reply["id"], "parent_comment_id": tid, "video_id": video_id,
                    "is_reply": True, "author": rs["authorDisplayName"],
                    "published_at": rs["publishedAt"], "updated_at": rs["updatedAt"],
                    "like_count": rs["likeCount"], "text": rs["textOriginal"],
                })

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break
        time.sleep(1)

    # 2) yt-dlp deep-reply pass
    scraped = _scrape_replies_yt_dlp(video_id)
    api_ids = {r["comment_id"] for r in api_rows}
    yt_by_id = {c.get("id"): c for c in scraped if c.get("id")}

    # repair API rows' parent where yt-dlp knows a real one
    for r in api_rows:
        yc = yt_by_id.get(r["comment_id"])
        if yc and yc.get("parent") not in (None, "root"):
            r["parent_comment_id"] = yc["parent"]
            r["is_reply"] = True

    # add yt-dlp-only comments
    for c in scraped:
        cid = c.get("id")
        if not cid or cid in api_ids:
            continue
        parent = c.get("parent")
        is_reply = parent not in (None, "root")
        ts = c.get("timestamp")
        published = (
            datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            if isinstance(ts, (int, float)) else None
        )
        api_rows.append({
            "comment_id": cid, "parent_comment_id": parent if is_reply else None,
            "video_id": video_id, "is_reply": is_reply,
            "author": c.get("author", ""), "published_at": published,
            "updated_at": None, "like_count": c.get("like_count", 0),
            "text": c.get("text", ""),
        })

    # 3) infer nested threading
    df = _infer_threading(pd.DataFrame(api_rows))
    return df.to_dict("records")

# Define the maximum number of videos/shorts
max_shorts = 100 
max_comments = 14400

# # TODO: TESTING SETUP
# max_shorts = 2
# max_comments = 50

### Setup directories
output_dir = "VideosComments/youtube"
output_video_info_dir = f"{output_dir}/metadata"
output_video_comments_dir = f"{output_dir}/comments"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_video_info_dir, exist_ok=True)
os.makedirs(output_video_comments_dir, exist_ok=True)

# Keep track of authors to anonymize across videos.
ANON_MAP_PATH = f"{output_dir}/author_anon_map.json"

# Load existing map (or start fresh)
if os.path.exists(ANON_MAP_PATH):
    with open(ANON_MAP_PATH) as f:
        author_anon_map = json.load(f)
else:
    author_anon_map = {}

def get_anon_id(author):
    if author not in author_anon_map:
        author_anon_map[author] = f"author_{len(author_anon_map)}"
    return author_anon_map[author]

# Filter newspapers based on the command-line argument
if args.newspaper_source != "all":
    newspapers = {k: v for k, v in newspapers.items() if k in args.newspaper_source.split(",")}

for key, accountname in newspapers.items():
    print(f"Processing channel: {accountname}")
    newspaper = key

    # Find shorts (>= 1 day old) from the channel, with full info already fetched
    shorts_entries = get_shorts_from_channel(accountname, newspaper, max_shorts=max_shorts)
    print(f"\tFound {len(shorts_entries)} shorts for {newspaper}")

    # Iterate through each short and fetch the comments
    for video_info in tqdm(shorts_entries, desc=f"Processing shorts for {newspaper}"):
        # Create newspaper specific directories
        os.makedirs(f"{output_video_info_dir}/{newspaper}", exist_ok=True)
        os.makedirs(f"{output_video_comments_dir}/{newspaper}", exist_ok=True)

        # video_info already fetched in get_shorts_from_channel — save directly
        video_id = video_info['video_id']
        with open(f"{output_video_info_dir}/{newspaper}/{video_id}.json", "w") as f:
            json.dump(video_info, f, indent=4)
        
        # Get comments and save as csv
        comments = fetch_comments(video_id, max_comments)  # List of dicts
        if not comments:
            print(f"  No comments for {newspaper} short {video_id} — skipping.")
            
            # Create empty DataFrame with expected columns to save as CSV
            empty_df = pd.DataFrame(columns=["comment_id", "parent_comment_id", "video_id", "is_reply", "author",
                                           "published_at", "updated_at", "like_count", "text",
                                           "newspaper", "author_anon"])
            empty_df.to_csv(f"{output_video_comments_dir}/{newspaper}/{video_id}.csv", index=False)
            print(f"  Saved empty comments CSV for {newspaper} short {video_id}.")
            continue

        df = pd.DataFrame(comments)
        if "author" not in df.columns:
            print(f"  Comments for {video_id} missing 'author' field — skipping.")
            empty_df = pd.DataFrame(columns=["comment_id", "parent_comment_id", "video_id", "is_reply", "author",
                                           "published_at", "updated_at", "like_count", "text",
                                           "newspaper", "author_anon"])
            empty_df.to_csv(f"{output_video_comments_dir}/{newspaper}/{video_id}.csv", index=False)
            print(f"  Saved empty comments CSV for {newspaper} short {video_id}.")
            continue
        df["newspaper"] = newspaper
        df["author_anon"] = df["author"].apply(get_anon_id)

        
        df.to_csv(f"{output_video_comments_dir}/{newspaper}/{video_id}.csv", index=False)
        print(f"Saved comments for {newspaper} short with video ID {video_id}. Shape: {df.shape}")

# Save the updated anon map
with open(ANON_MAP_PATH, "w") as f:
    json.dump(author_anon_map, f, indent=4)