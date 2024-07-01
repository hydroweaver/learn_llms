# #Lot of stuff when looking at other embedding models, just trying openai embedding model stuff listed at https://platform.openai.com/docs/guides/embeddings

# #not running a local model, but sending stuff to openai embedding enpoint
import json
import requests
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path='C:/Users/hydro/Downloads/projects/learn_llms/.env') 

print(str(os.getenv('OPENAI_API_KEY')))


# def req():
#     r = requests.post("https://api.openai.com/v1/embeddings",
#               data=json.dumps({
#                   "input": "The food was delicious and the waiter...",
#                   "model": "text-embedding-ada-002",
#                   "encoding_format": "float"}),
#                 headers={
#                   "Authorization" : "Bearer " + str(os.getenv('OPENAI_API_KEY')),
#                   "Content-Type": "application/json"
#               })
#     print(r.text)

# req()

# # # Embedding model output

# # # Rag embedding relationship vector db

# # from transformers import AutoModel
# # from numpy.linalg import norm

# # cos_sim = lambda a,b: (a @ b.T) / (norm(a)*norm(b))
# # model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True) # trust_remote_code is needed to use the encode method
# # embeddings = model.encode(['How is the weather today?', 'What is the current weather like today?'])
# # print(cos_sim(embeddings[0], embeddings[1]))
# # print(embeddings[0])