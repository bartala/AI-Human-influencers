#Import from C:\seminar
#create df
import pandas as pd
import json
import os
from dotenv import load_dotenv


start_index = 150000  #if crashed then check what the latest json and write the count
#df = pd.read_excel('.../All_users_YES_POSTURL_14.6.24_RELEVANT COPY.xlsx')
#df = pd.read_json(f"OutPutUntil{start_index}.json")
df = pd.read_json(f"output_Final.json")

#embedding
import openai


# Set up your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

import time
import numpy as np
#from openai.error import ServiceUnavailableError

def get_embedding(text):

    if text is None:
        text = ""
    elif not isinstance(text, str):
        text = str(text)

    # Escape special characters to ensure valid JSON
    text = json.dumps(text)

    response = openai.Embedding.create(
        model="text-embedding-3-large",
        input=text
    )
    embedding = response['data'][0]['embedding']
    return embedding

def get_embedding_with_retry(text, retries=5, delay=5):
    for i in range(retries):
        try:
            return get_embedding(text)
        except Exception:
            if i < retries - 1:
                time.sleep(delay)
            else:
                print(text + ' got a error')


#create column
# slice by index position
#sliced_df = df.iloc[0:25000]

# To print the index, you can add a print statement inside the lambda
#try:
#  df['embedded.posts.comments.text'] = df.apply(
#      lambda row: print(f"Processing index: {row.name}") or np.array(get_embedding_with_retry(row['posts.comments.text'],row,df)), axis=1)
#except Exception:
#    print("Service Unavailable")

if(start_index == 0):
    df['embedded.posts.comments.text'] = None

try:
    for index, row in df.iloc[start_index:].iterrows():
        print(f"Processing index: {index}")
        embedding = get_embedding_with_retry(row['posts.comments.text'])
        if embedding is not None:
           # embedding_str = np.array(embedding).astype(str).tolist()
            df.at[index, 'embedded.posts.comments.text'] = embedding
        else:
            df.at[index, 'embedded.posts.comments.text'] = None
        if (index % 25000 == 0 and index != 0):
            df.to_json(f"OutPutUntil{index}.json", index=False)
except Exception as ex:
    print(ex)

#export file
df.to_json('output_Final.json', index=False)
#Export files to C:\your_dir...





