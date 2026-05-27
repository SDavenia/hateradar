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
    "repubblica": "@repubblica"
}

# Replace with your own API Key
load_dotenv()
DEVELOPER_KEY = os.getenv("youtube_api")

# Initialize YouTube API client
youtube = googleapiclient.discovery.build(
    "youtube", "v3", developerKey=DEVELOPER_KEY)


def get_shorts_from_channel(channel_url):
    """
    Extract all Shorts URLs from a YouTube channel.
    """
    MAX_SHORTS = 2
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
            if len(shorts) >= MAX_SHORTS:
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
    comments = []
    next_page_token = None

    while len(comments) < max_comments:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            # maxResults=100,  # Max is 100
            maxResults = 10, # TODO: Only for now
            pageToken=next_page_token  # Use the next page token
        )
        response = request.execute()

        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']
            ids = item['snippet']['topLevelComment']['id']
            comments.append([
                ids, 
                comment['authorDisplayName'],
                comment['publishedAt'],
                comment['updatedAt'],
                comment['likeCount'],
                comment['textDisplay']
            ])

        # Check if next page is available
        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break

        time.sleep(1)
    return comments[:max_comments]

for key, accountname in newspapers.items():
    newspaper = key
    # Find 100 most recent shorts from the channel
    shorts_entries = get_shorts_from_channel(accountname)
    print(f"Found {len(shorts_entries)} shorts for {newspaper}")

    # Iterate through each short and fetch the comments
    max_comments = 14400
    max_comments = 10 # TODO: Only for now 
    for entry in tqdm(shorts_entries, desc=f"Processing shorts for {newspaper}"):
        video_id = entry['video_id']
        comments = fetch_comments(video_id, max_comments)
        df = pd.DataFrame(comments, columns=['id','author', 'published_at', 'updated_at', 'like_count', 'text'])
        df["newspaper"] = newspaper
        df["author_anon"] = df["author"].astype("category").cat.codes.map(lambda x: f"author_{x}") #--> ATTENZIONE! Gli autori sono anonimizzati per df. Da cambiare se si usa un df unico. 
        os.makedirs("./VideosComments/youtube", exist_ok=True)
        df.to_csv(f"./VideosComments/youtube/youtube_comments_{newspaper}_{video_id}.csv", index=False)
        print(f"Saved comments for {newspaper} short with video ID {video_id}. Shape: {df.shape}")