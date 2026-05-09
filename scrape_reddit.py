import os
import time
import praw
import json
import pandas as pd
from dotenv import load_dotenv
from urllib.parse import urlparse, urlunparse
from prawcore.exceptions import TooManyRequests


SUBREDDITS_GENERALI = [
    "italy", "italia", "ItaliaRossa", "politicaITA", "oknotizie", "EconomiaItaliana", "innovazione" 
]

SUBREDDITS_REGIONALI = [
    "Abruzzo", "Basilicata", "Calabria", "Emilia_Romagna", "Friuli", "Liguria",
    "Lombardia", "Marche", "Molise", "Piemonte", "Puglia", "Sardegna", "Sicilia", 
    "Toscana", "Trentino_alto_Adige", "Umbria", "Veneto"
]

SUBREDDITS_LOCALI = [
    "Bologna", "Firenze", "Milano", "Napoli", "Padova", "Roma", "Torino", "Modena", 
    "Trieste", "Genova", "Bari", "Catania", "Siracusa", "Trento", "Perugia", "Aosta", 
    "Venezia", "brescia", "cagliari", "parma"
]

def search_subreddits(
    subreddits_str,
    df,
    reddit,
    time_filter="month",
    limit=50,
    max_retries=5,
    base_sleep=30):

    results = []
    total_len = len(df)
    for idx, row in df.iterrows():

        url = row["clean_urls"]
        query = f"url:{url}"
        success = False

        for attempt in range(max_retries):
            try:
                submissions = reddit.subreddit(subreddits_str).search(
                    query,
                    sort="new",
                    time_filter=time_filter,
                    limit=limit
                )
                submissions = list(submissions)
                for submission in submissions:
                    print(
                        f"\tSubmission subreddit: {submission.subreddit}, "
                        f"title: {submission.title}, "
                        f"permalink: {submission.permalink}"
                    )
                    results.append(submission)
                success = True
                # # added a small pause between normal requests to try and avoid the rate limit
                # time.sleep(2)
                break 

            except TooManyRequests as e:
                # Use Reddit retry_after if available
                retry_after = getattr(e, "retry_after", None)
                if retry_after is not None:
                    wait_time = retry_after
                else:
                    wait_time = base_sleep * (2 ** attempt) #Maybe too long, see later TODO
                print(
                    f"429 TooManyRequests for {url} "
                    f"(attempt {attempt + 1}/{max_retries}) "
                    f"-> sleeping {wait_time}s"
                )
                time.sleep(wait_time)

            except Exception as e:
                print(f"Error for {url}: {e}")
                time.sleep(5)
                break

        if not success:
            print(f"FAILED after {max_retries} retries: {url}")
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{total_len} URLs")

    return results



def extract_submission_data(results, max_depth=10):
    final_results = []
    for submission in results:
        output_json = {}
        # General info
        output_json["id"] = submission.id
        output_json["subreddit"] = str(submission.subreddit)
        output_json["permalink"] = submission.permalink
        output_json["created_utc"] = submission.created_utc

        output_json["author"] = str(submission.author)


        output_json["title"] = submission.title
        output_json["selftext"] = submission.selftext
        output_json["url"] = submission.url

        output_json["comment"] = [extract_comment(comment, max_depth=max_depth) for comment in submission.comments]
        final_results.append(output_json)
    return final_results


def normalize_url(u):
    try:
        parsed = urlparse(u.lower())

        # remove query params + fragments
        clean = parsed._replace(
            query="",
            fragment=""
        )

        path = clean.path.rstrip("/")

        # optional: remove .html
        path = path.replace(".html", "")

        clean = clean._replace(path=path)

        return urlunparse(clean)

    except Exception:
        return u.lower()

def extract_comment(comment, depth=0, max_depth=10):
    """Recursively extract comment tree"""
    try:
        if depth > max_depth:
            return None

        return {
            "id": comment.id,
            "author": str(comment.author),
            "body": comment.body,
            "score": comment.score,
            "created_utc": comment.created_utc,
            "replies": [
                child for child in (
                    extract_comment(reply, depth + 1, max_depth)
                    for reply in comment.replies
                ) if child is not None
            ],
        }
    except Exception as e:
        return {
            "id": getattr(comment, "id", None),
            "error": str(e)
        }


def setup_reddit_client():
    CLIENT_ID = os.getenv("client_id")
    CLIENT_SECRET = os.getenv("client_secret")

    reddit = praw.Reddit(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        user_agent="script:italy_url_tracker:v1.0",
    )
    return reddit

def main():
    load_dotenv()
    # Load All jsonl files:
    data_dir = "data/202604_21-28/"
    output_data_dir = "data/outputs/202404_21-28/"
    all_jsonl_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".jsonl")]

    # Setup reddit client
    reddit = setup_reddit_client()

    # Read from subreddits (all_subreddits_str)
    subreddits_generali_str = "+".join(SUBREDDITS_GENERALI)
    subreddits_regionali_str = "+".join(SUBREDDITS_REGIONALI)
    subreddits_locali_str = "+".join(SUBREDDITS_LOCALI)
    all_subreddits_str = "+".join([subreddits_generali_str, subreddits_regionali_str, subreddits_locali_str])

    # Process each file:
    print(f"Processing {len(all_jsonl_paths)} files...")
    for idx, file_path in enumerate(all_jsonl_paths):
        print(f"Processing file {idx + 1}/{len(all_jsonl_paths)}: {file_path}")
        # Extract date from filename
        # file is called with the date outptu_20260421.jsonl
        date = os.path.basename(file_path).split("_")[1].split(".")[0]
        date = f"{date[:4]}-{date[4:6]}-{date[6:]}" # convert to YYYY-MM-DD
        
        output_path = os.path.join(output_data_dir, f"reddit_results_{date}.jsonl")
        print(f"Output will be saved to: {output_path}")

        with open(file_path, "r") as f:
            data = [json.loads(line) for line in f]
        df = pd.DataFrame(data)
        # Remove zazzom
        df = df[df["MentionSourceName"] != "zazoom.it"].reset_index(drop=True)

        df[date] = [date] * len(df)
        df['date'] = pd.to_datetime(df['date'])
        # Clean urls
        df['clean_urls'] = df['MentionIdentifier'].apply(normalize_url)
        
        # Keep only urls to search for:
        df_forsearch = df[["clean_urls"]].drop_duplicates().reset_index(drop=True)

        results_generali = search_subreddits(all_subreddits_str,
                                            df_forsearch,
                                            reddit,
                                            time_filter="month",
                                            limit=20,
                                            max_retries=3,
                                            base_sleep=30)

        print(f"Found {len(results_generali)} submissions for subreddits: {all_subreddits_str}")
        final_results_generali = extract_submission_data(results_generali, max_depth=10)
        
        # Add back global event id, title and text from the original df
        enriched_results = []

        for submission in final_results_generali:
            url = submission["url"]
            matching_rows = df[df["clean_urls"] == url]
            enriched_submission = submission.copy()

            if not matching_rows.empty:
                enriched_submission["GlobalEventID"] = matching_rows["GlobalEventID"].tolist()
                # Those below are unique so we can just take the first one.
                enriched_submission["MentionSourceName"] = matching_rows["MentionSourceName"].tolist()[0]
                enriched_submission["Title"] = matching_rows["Title"].tolist()[0]
                enriched_submission["Text"] = matching_rows["Text"].tolist()[0]

            enriched_results.append(enriched_submission)

        with open(output_path, "w") as f:
            for item in enriched_results:
                f.write(json.dumps(item) + "\n")
        

if __name__ == "__main__":
    main()