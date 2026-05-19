import json
import os
from glob import glob
import pandas as pd
import re


"""
Processes Reddit post data from JSONL files into a flat pandas DataFrame.

considered keys: 
post
dict_keys(['id', 'subreddit', 'permalink', 'created_utc', 'author', 'title', 'selftext', 'url', 'comment'])

comment
dict_keys(['id', 'author', 'body', 'score', 'created_utc', 'replies'])


Each row represents a single comment (or a post with no comments).
Replies are recursively disaggregated, preserving thread structure via:
- comment_level: depth of the comment (0 = top-level, 1 = reply, 2 = reply to reply, ...)
- parent_id:     id of the comment being replied to (None for top-level comments)
- thread_id:     id of the root comment of the thread

Posts with no comments are kept as a single row with all comment fields set to None.
Source file date is extracted from the filename and added as a column.
"""


dir_reddit = './data/outputs/202404_21-28/'
rows = []
post_ids = list()

def extract_comments(comments, post_data, level=0, parent_id=None, thread_id=None):
    for comment in comments:
        cid = comment.get('id')
        root = cid if level == 0 else thread_id

        row = {
            **post_data,
            'comment_id':      cid,
            'comment_author':  comment.get('author'),
            'comment_body':    comment.get('body'),
            'comment_score':   comment.get('score'),
            'comment_created': comment.get('created_utc'),
            'comment_level':   level,
            'parent_id':       parent_id,
            'thread_id':       root,
        }
        rows.append(row)

        replies = comment.get('replies', [])
        if replies:
            extract_comments(replies, post_data, level=level + 1, parent_id=cid, thread_id=root)

for file in glob(os.path.join(dir_reddit, '*.jsonl')):
    basename = os.path.basename(file)
    date_str = re.search(r'\d{4}-\d{2}-\d{2}', basename).group()

    with open(file, 'r') as f:
        for line in f:
            post = json.loads(line)

            post_data = {
                'post_id':      post['id'],
                'subreddit':    post['subreddit'],
                'permalink':    post['permalink'],
                'post_created': post['created_utc'],
                'post_author':  post['author'],
                'title':        post['title'],
                'selftext':     post['selftext'],
                'url':          post['url'],
                'source_file':  date_str,
            }

            post_ids.append(post['id'])

            comments = post.get('comment', [])
            if comments:
                extract_comments(comments, post_data, level=0, parent_id=None, thread_id=None)
            else:
                rows.append({
                    **post_data,
                    'comment_id':      None,
                    'comment_author':  None,
                    'comment_body':    None,
                    'comment_score':   None,
                    'comment_created': None,
                    'comment_level':   None,
                    'parent_id':       None,
                    'thread_id':       None,
                })

df = pd.DataFrame(rows)

print(df.shape)
df.to_csv('./data/dataset/reddit_data.csv', index=False)

# statistics
print("Total post IDs from jsonl files:", len(post_ids), "--- Unique:", len(set(post_ids)))
print('\n\n # DF Statistics # \n')
print("Total posts:", df['post_id'].nunique())
print("Total comments:", df['comment_id'].nunique())
print("Average comments per post:", df.groupby('post_id')['comment_id'].nunique().mean())
print("Max comment depth:", df['comment_level'].max())


for i in df['post_id']:
    if i not in post_ids:
        print(i)
