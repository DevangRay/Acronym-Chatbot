import openai as openai
import pandas as pd
import tiktoken
import matplotlib
import numpy as np
from openai.embeddings_utils import distances_from_embeddings


openai.api_key = "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"

df = pd.read_excel('definitions_abbreviations.xlsx')
print(df.head())

# Load the cl100k_base tokenizer which is designed to work with the ada-002 model
tokenizer = tiktoken.get_encoding("cl100k_base")
df.columns = ['Term','Definition','Relation to','Description']
print(df['Definition'].astype(str))

# Tokenize the text and save the number of tokens to a new column
df['n_tokens'] = df['Definition'].astype(str).apply(lambda x: len(tokenizer.encode(x)))

df.n_tokens.hist()


max_tokens = 500
def split_into_many(text, max_tokens = max_tokens):

    # Split the text into sentences
    sentences = text.split('. ')

    # Get the number of tokens for each sentence
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]

    chunks = []
    tokens_so_far = 0
    chunk = []

    # Loop through the sentences and tokens joined together in a tuple
    for sentence, token in zip(sentences, n_tokens):

        # If the number of tokens so far plus the number of tokens in the current sentence is greater
        # than the max number of tokens, then add the chunk to the list of chunks and reset
        # the chunk and tokens so far
        if tokens_so_far + token > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            chunk = []
            tokens_so_far = 0

        # If the number of tokens in the current sentence is greater than the max number of
        # tokens, go to the next sentence
        if token > max_tokens:
            continue

        # Otherwise, add the sentence to the chunk and add the number of tokens to the total
        chunk.append(sentence)
        tokens_so_far += token + 1

    return chunks


shortened = []

# Loop through the dataframe
for row in df.iterrows():

    # If the text is None, go to the next row
    if row[1]['Definition'] is None:
        continue

    # If the number of tokens is greater than the max number of tokens, split the text into chunks
    if row[1]['n_tokens'] > max_tokens:
        shortened += split_into_many(row[1]['Definition'])

    # Otherwise, add the text to the list of shortened texts
    else:
        shortened.append( row[1]['Definition'] )

df = pd.DataFrame(shortened, columns = ['Definition'])
df['n_tokens'] = df['Definition'].astype(str).apply(lambda x: len(tokenizer.encode(x)))
df.n_tokens.hist()


# df['embeddings'] = df['Definition'].astype(str).apply(lambda x: openai.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])
# df.to_csv('embeddings.csv')
# df.head()





df=pd.read_csv('embeddings.csv', index_col=0)
df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)

df.head()


