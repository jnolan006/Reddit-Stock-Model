!pip install --ignore-installed blinker --quiet
!pip install requests --quiet
!pip install beautifulsoup4 --quiet
!pip install alpaca-trade-api==3.2.0 --quiet
!pip install lumibot timedelta alpaca-trade-api --quiet
!pip install lumibot==2.9.13 --quiet
!pip install praw --quiet
!pip install torch torchvision torchaudio transformers --quiet

import requests
from bs4 import BeautifulSoup
import pandas as pd
import pandas as pd
import requests
from bs4 import BeautifulSoup
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Tuple
import logging

import psycopg2
import pandas as pd
from datetime import datetime, timedelta
from datetime import date, timedelta
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score
from sklearn.tree import _tree
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import pickle
from sklearn.preprocessing import StandardScaler
from itertools import combinations

import requests, time, re, os
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl
import numpy as np
import datetime

from scipy import linalg
import math
from datetime import datetime

import time
from datetime import datetime
import os
import sys
import pickle

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

host = "ec2-52-6-117-96.compute-1.amazonaws.com"
dbname = "dftej5l5m1cl78"
user = "aiuhlrpcnftsjs"
password = "8b2220cd5b6da572369545d91f6b435dfc37a42bfec6b6e2a5c9f236dfb65f42"

conn = psycopg2.connect(host=host, dbname=dbname, user=user, password=password)
cur = conn.cursor()

from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from datetime import datetime
from alpaca_trade_api import REST
from timedelta import Timedelta
import time

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import praw
import matplotlib.pyplot as plt
import math
import datetime as dt
import pandas as pd
import numpy as np
import csv
import re

import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

reddit = praw.Reddit(client_id='F7eG_5Prpu3lj9Xz9CeDLg',
                    client_secret='aijmI4ZwbR4b2eqJpIZLtEbewm1H5Q',
                    user_agent='jnolan006',
                    check_for_async=False)

subreddits = ["wallstreetbets", "stocks", "investing", "options", "StockMarket", "pennystocks", "RobinHood", "UndervaluedStonks", "InvestmentClub", "EducatedInvesting", "smallstreetbets", "stonks", "Wallstreetbetsnew", "investing_discussion"]


def get_sentiment_reddit(ticker, urlT):
    subComments = []
    bodyComment = []
    try:
        check = reddit.submission(url=urlT)
        subComments = check.comments
    except:
        return 'NA', 0, [], 0  

    for comment in subComments:
        try:
            bodyComment.append(comment.body)
        except:
            continue  

    if bodyComment:
        inputs = tokenizer(bodyComment, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        positive_probability = probabilities[:, 1].mean().item()
        negative_probability = probabilities[:, 0].mean().item()
        if positive_probability > negative_probability:
            sentiment = "positive"
            probability = positive_probability
        else:
            sentiment = "negative"
            probability = negative_probability
    else:
        sentiment = 'NA'
        probability = 0
    num_comments = len(bodyComment)
    return sentiment, probability, bodyComment, num_comments

def scrape_yahoo_trending_tickers(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        symbol_list = []
        for link in soup.find_all('a'):
            symbol = (link.get('href'))
            if '/quote/' in symbol and '%' not in symbol:
                split_html = symbol.split('/')
                symbol = split_html[-1]
                symbol_list.append(symbol)

        return symbol_list

yahoo_trending_tickers_url = 'https://finance.yahoo.com/trending-tickers'
test = scrape_yahoo_trending_tickers(yahoo_trending_tickers_url)
symbol_list = []
for symbol in test:
    symbol_list.append(symbol)
print(symbol_list)

for ticker in symbol_list:
    url = f'https://finance.yahoo.com/quote/{ticker}/community?p=TSL{ticker}'
    print(url)
    response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/110.0'})
    soup = BeautifulSoup(response.text, 'html.parser')
    data = json.loads(soup.select_one('#spotim-config').get_text(strip=True))['config']

    url = "https://api-2-0.spot.im/v1.0.0/conversation/read"
    payload = json.dumps({
      "conversation_id": data['spotId'] + data['uuid'].replace('_', '$'),
      "count": 250,
      "offset": 0
    })
    headers = {
      'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/110.0',
      'Content-Type': 'application/json',
      'x-spot-id': data['spotId'],
      'x-post-id': data['uuid'].replace('_', '$'),
    }

    response = requests.post(url, headers=headers, data=payload)
    data = response.json()

    bullish_count = 0
    bearish_count = 0

    for comment in data['conversation']['comments']:
        if 'additional_data' in comment and 'labels' in comment['additional_data'] and 'ids' in comment['additional_data']['labels']:
            labels = comment['additional_data']['labels']['ids']
            if 'BULLISH' in labels:
                bullish_count += 1
            if 'BEARISH' in labels:
                bearish_count += 1

    total_submissions_pos = 0
    total_comments_pos = 0
    total_score_pos = 0
    total_submissions_neg = 0
    total_comments_neg = 0
    total_score_neg = 0
    for subreddit_name in subreddits:
        subreddit = reddit.subreddit(subreddit_name)
        for submission in subreddit.search(f'selftext:{ticker}', time_filter='week'):
            if submission.subreddit.display_name not in subreddits:
                continue
            if submission.created_utc >= start_date.timestamp():
                urlT = submission.url
                print(urlT)
                if urlT:
                    sentiment, probability, bodyComment, num_comments = get_sentiment_reddit(ticker, urlT)
                    print(sentiment)
                    print(probability)
                    print(bodyComment)
                    print(num_comments)
                else:
                    sentiment = None
                    probability = 0
                    num_comments = 0
                if sentiment == "positive":
                    total_submissions_pos += 1
                    total_comments_pos += num_comments
                    total_score_pos += probability
                if sentiment == "negative":
                    total_submissions_neg += 1
                    total_comments_neg += num_comments
                    total_score_neg += probability
                time.sleep(1)
    if total_submissions_pos > 0:
        average_score_pos = total_score_pos / total_submissions_pos
    else:
        average_score_pos = 0
    score_pos = total_submissions_pos * total_comments_pos * average_score_pos

    if total_submissions_neg > 0:
        average_score_neg = total_score_neg / total_submissions_neg
    else:
        average_score_neg = 0
    score_neg = total_submissions_neg * total_comments_neg * average_score_neg

    time.sleep(1)

    df = pd.DataFrame({
        'Symbol': [ticker],
        'Bullish_Count': [bullish_count],
        'Bearish_Count': [bearish_count],
        'Reddit Score Pos': [score_pos],
        'Reddit Score neg': [score_neg]
    })

    print(df)
