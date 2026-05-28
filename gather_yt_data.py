# import requests, yaml
import argparse
import pandas as pd
import time
from tqdm import tqdm
import googleapiclient.discovery
import time
import os
from dotenv import load_dotenv
import re
import json
import time
import pandas as pd
from yt_dlp import YoutubeDL
from youtube_comment_downloader import YoutubeCommentDownloader
from youtube_comment_downloader import SORT_BY_POPULAR

parser = argparse.ArgumentParser()

parser.add_argument('-c','--config')
args = parser.parse_args()


# with open(args.config) as f:
#     vars = yaml.load(f, Loader=yaml.FullLoader)


# celebrity_id = {
#     "J K Rowling": "fNhGH-EBKfM",
#     "Kanye West": "Wz9bRGfQG5g", 
#     "Lizzo": "RRCTASidiM8",
#     "Halle Bailey": "lVAkzXp4xNU", 
#     "Ellen DeGeneres": "MZuvETZg1rA",
#     "Andrew Tate": "xfqtOUC8xqY"
#     }
GENERAL_CHANNEL_URL = "https://www.youtube.com/@{CHANNEL_NAME}"

newspapers = {
    "repubblica": "@repubblica",
    "corriere_della_sera": "@CorrieredellaSera",
    "la_gazzetta_dello_sport": "@LaGazzettadelloSport",
    "il_sole_24_ore": "@ilsole24ore",
    "avvenireNEI": "@AvvenireNEI",
    "lastampa": "@LaStampa",
    "il_fatto_quotidiano": "@IlFattoQuotidiano",
    "ilmessaggero": "@ilmessaggero",
    "il_quotidiano_nazionale": "@quotidianonet",
    "il_gazzettino": "@ilgazzettino",
    "il_giornale": "@ilgiornale"
}

# Replace with your own API Key
load_dotenv()
DEVELOPER_KEY = os.getenv("youtube_api")

# Initialize YouTube API client
youtube = googleapiclient.discovery.build(
    "youtube", "v3", developerKey=DEVELOPER_KEY)

def fetch_video_info(video_id, newspaper):
    """
    Get video related information for each video.
    """
    request = youtube.videos().list(
        part="snippet,statistics",
        id=video_id
    )
    response = request.execute()

    if not response['items']:
        return None

    print(response['items'])

    item     = response['items'][0]
    snippet  = item['snippet']
    stats    = item['statistics']

    return {
        'video_id':      video_id,
        'newspaper':     newspaper,
        'title':         snippet['title'],
        'channel':       snippet['channelTitle'],
        'published_at':  snippet['publishedAt'],
        'description':   snippet.get('description', ''),
        'tags':          ', '.join(snippet.get('tags', [])),  # flattened for CSV
        'view_count':    int(stats.get('viewCount', 0)),
        'like_count':    int(stats.get('likeCount', 0)),      # 0 if likes hidden
        'comment_count': int(stats.get('commentCount', 0)),
    }


def get_shorts_from_channel(channel_url, max_shorts=100):
    """
    Extract all Shorts URLs from a YouTube channel.
    """
    channel_url = GENERAL_CHANNEL_URL.format(CHANNEL_NAME=channel_url.lstrip("@"))
    shorts_url = channel_url.rstrip("/") + "/shorts"

    ydl_opts = {
        "extract_flat": True,
        "quiet": True,
        "skip_download": True,
    }

    shorts = []

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(shorts_url, download=False)
        print(f"Info extracted for {shorts_url}")

        if "entries" not in info:
            print("No shorts found.")
            return shorts

        for entry in info["entries"]:
            if len(shorts) % 10 == 0:
                print(f"Found {len(shorts)} shorts so far...")
            if not entry:
                continue
            if len(shorts) >= max_shorts:
                break

            video_id = entry.get("id")
            title = entry.get("title")

            if video_id:
                shorts.append({
                    "video_id": video_id,
                    "title": title,
                    "url": f"https://www.youtube.com/shorts/{video_id}"
                })

    return shorts

def fetch_comments(video_id, max_comments):
    """
    Fetch comments for a given video ID, limited by youtube data API to only first reply for each video
    """
    comments = []
    next_page_token = None

    while len(comments) < max_comments:
        request = youtube.commentThreads().list(
            part="snippet,replies",
            videoId=video_id,
            maxResults=100,  # Max is 100
            pageToken=next_page_token  # Use the next page token
        )
        response = request.execute()

        # Iterate through each first level comment (top_comment) and extract all second-level comments in sub_comments field.
        for item in response['items']:
            top_comment = item['snippet']['topLevelComment']
            top_comment_id = top_comment['id']
            top_comment_snippet = item['snippet']['topLevelComment']['snippet']
            
            # top_comment = item['snippet']['topLevelComment']['snippet']
            # ids = item['snippet']['topLevelComment']['id']
            # top_comment_id = item['id']

            comments.append({
                'comment_id': top_comment_id,
                'parent_comment_id': None,
                'video_id': video_id,
                'is_reply': False,
                'author': top_comment_snippet['authorDisplayName'],
                'published_at': top_comment_snippet['publishedAt'],
                'updated_at': top_comment_snippet['updatedAt'],
                'like_count': top_comment_snippet['likeCount'],
                'text': top_comment_snippet['textOriginal']
            })

            replies = item.get('replies', {}).get('comments', [])
            
            for reply in replies:
                reply_snippet = reply['snippet']

                comments.append({
                    'comment_id': reply['id'],
                    'parent_comment_id': top_comment_id,
                    'video_id': video_id,
                    'is_reply': True,
                    'author': reply_snippet['authorDisplayName'],
                    'published_at': reply_snippet['publishedAt'],
                    'updated_at': reply_snippet['updatedAt'],
                    'like_count': reply_snippet['likeCount'],
                    'text': reply_snippet['textOriginal']
                })

        # Check if next page is available
        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break

        time.sleep(1)
    return comments[:max_comments]

# Define the maximum number of videos/shorts
max_shorts = 100 
max_comments = 14400

# TODO: TESTING SETUP
max_shorts = 2
max_comments = 15

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

for key, accountname in newspapers.items():
    print(f"Processing channel: {accountname}")
    newspaper = key
    # Find 100 most recent shorts from the channel
    shorts_entries = get_shorts_from_channel(accountname, max_shorts=max_shorts)
    print(f"\tFound {len(shorts_entries)} shorts for {newspaper}")

    # Iterate through each short and fetch the comments
    for entry in tqdm(shorts_entries, desc=f"Processing shorts for {newspaper}"):
        # Create newspaper specific directories
        os.makedirs(f"{output_video_info_dir}/{newspaper}", exist_ok=True)
        os.makedirs(f"{output_video_comments_dir}/{newspaper}", exist_ok=True)

        # Get video info and save as json
        video_id = entry['video_id']
        video_info = fetch_video_info(video_id, newspaper)
        if video_info is None:
            continue
        # Save to video_info dir as json
        with open(f"{output_video_info_dir}/{newspaper}/{video_id}.json", "w") as f:
            json.dump(video_info, f, indent=4)

        # Get comments and save as csv
        comments = fetch_comments(video_id, max_comments) # List of dicts 
        df = pd.DataFrame(comments)
        df["newspaper"] = newspaper
        df["author_anon"] = df["author"].apply(get_anon_id)
        df.to_csv(f"{output_video_comments_dir}/{newspaper}/{video_id}.csv", index=False)
        print(f"Saved comments for {newspaper} short with video ID {video_id}. Shape: {df.shape}")
    
    # TODO: Remove, break for now -> Only first 2 newspapers 
    break

# Save the updated anon map
with open(ANON_MAP_PATH, "w") as f:
    json.dump(author_anon_map, f, indent=4)