import pandas as pd
import json
import torch
import os
import asyncio

from searchAPIassist import searchInternet
from askLLM import ask_llm

from langchain_text_splitters import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer, util, CrossEncoder

# Process and clean the corpus data from the csv file
def process_csv_lineByline(file):
    rows = []
    data = pd.read_csv(file, engine="python", encoding='utf-8', escapechar='\\', on_bad_lines='skip')
    data = data["Answer"]

    for line in data:
        line = str(line).strip()
        if not line:
            continue
                
        if len(line) >= 2 and line[0] == '"' and line[-1] == '"':
            line = line[1:-1]
        
        if line.replace(".","", 1).isdigit():
            continue

        if line == "Answer":
            continue

        rows.append(line)
    return pd.DataFrame({"answers": rows})
        

corpus = process_csv_lineByline('0000.csv')

# queries = pd.read_csv('0000.csv')
# queries = queries['Question']

MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]

# Initialise the parameters for the text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
    separators=MARKDOWN_SEPARATORS,
    strip_whitespace=True
)

chunks, chunk_source = [], []

# Split the corpus into chunks and get the chunk source as well
for row, text in enumerate(corpus['answers']):
    text = str(text).strip()
    if not text:
        continue
    text = splitter.split_text(text)
    chunks.extend(text)
    chunk_source.extend(len(text) * [row])

# Initialise the encoders
biEncoder = SentenceTransformer("all-mpnet-base-v2")
crossEncoder = CrossEncoder('BAAI/bge-reranker-v2-m3', trust_remote_code=True)

# Embedding the chunks
EMBpath = 'chunk_embedding.pt'

if os.path.exists(EMBpath):
    chunkEmbeddings = torch.load(EMBpath)
else:
    chunkEmbeddings = biEncoder.encode(chunks, batch_size=256, show_progress_bar=True, normalize_embeddings=True, convert_to_tensor=True)
    torch.save(chunkEmbeddings, EMBpath)

# Retrieve the chunks for the RAG
def retrieve_rag_chunks(query, top_k):
    queryEmbedding = biEncoder.encode(query, convert_to_tensor=True)

    # Perform cosine similarity
    hits = util.semantic_search(queryEmbedding, chunkEmbeddings, top_k=200)[0]

    # Gather the chunks
    candidates = [chunks[hit['corpus_id']] for hit in hits[:50] if hit["corpus_id"] < len(chunks)]
    pairs = [[query, c] for c in candidates]

    # Get the score of the chunks and rank them
    scores = crossEncoder.predict(pairs)

    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)

    # Display the content and the score for similarity it got
    scoreRag = 0
    for score, content in ranked[:top_k]:
        scoreRag = score * 100
        print(f"Score: {scoreRag:.4f}\nContent: {content}\n")
    # print(f"Score: {ranked[0][1]:.4f}\nContent: {ranked[0][0]}\n")

    # Display the LLM answer if the below threshold -> fallback method
    if scoreRag < 80:
        print("answer from LLM:")
        print(asyncio.run(ask_llm(query)), "\n\n")

    # Search the internet if the below threshold -> fallback method
    if scoreRag < 80:
        answer = searchInternet(query)
        print(answer)
    else: 
        answer = ""

    # update the embeddings with the findings from the internet
    if answer != "":
        if os.path.exists(EMBpath):
            existEmbedding = torch.load(EMBpath)
            answerEmbed = biEncoder.encode(answer, convert_to_tensor=True)
            if answerEmbed.dim() == 1:
                answerEmbed = answerEmbed.unsqueeze(0)
            updateEmbed = torch.cat((existEmbedding, answerEmbed), dim=0)
            torch.save(updateEmbed, EMBpath)
            print("New embeddings uploaded!")
    else:
        print("No new embeddings!")

    return ranked[:top_k], answer

retrieve_rag_chunks("what is the speed of light", top_k=1)
