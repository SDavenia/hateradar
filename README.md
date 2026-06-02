# Towards an Offensive-Radar

This repository contains the code to replicate the experiments from the 

The full pipeline consists of a series of steps which are briefly described below.
## Repo structure

```
.
├── VideosComments/youtube  
├──── annotated_comments
├──── annotated_metadata   
├── create_gold.py
├── create_silver_comments.py
├── create_silver_metadata.py
├── gather_yt_data_time_nested.py
├── hateradar.py
├── prepare_hateradar_data.py
├── README.md
├── validate_silver_comments.py
└── validate_silver_metadata.py
```
Inside `annotated_comments`,`annotated_metadata` we have all files under `{newspaper}/{newspaper_video_id.csv/json}` respectively.

## Full Pipeline Description
The various scripts to obtain the full pipeline are briefly described here.
This code requires a Youtube data API key and a Huggingface Token to be placed inside a `.env` file, for scraping and modelling respectively.

0. `gather_yt_data_nested.py`
Code to scrape Youtube videos and reconstruct the full comment threads.

1. `create_gold.py`
Code to aggregated annotations to produce `_gold` files which are placed in `annotated_comments, annotated_metadata` folders with `_gold` suffix.

2. `validate_silver_comments.py, validate_silver_metadata.py`
Code to use the gold data to select the best model on offensiveness classification and topic classification. 
The best model is selected on a dev split consisting of 60\% of gold videos and performance is reported on a held-out 40\% test set.

3. `create_silver_comments.py, create_silver_metadata.py`
Code to infer silver labels for both offensive classification (at the comment level) and topic classification (at the news article level).

4. `prepare_offensiveradar_data.py`
Code to collect all gold and silver data and place them in a single `.csv` file containing `video_id, newspaper, text, topic, percentage_offensive_comments, type`, with one entry per video.

5. `offensiveradar.py`
Code to run zero-shot classification on the Offensive-Radar Detection task.
