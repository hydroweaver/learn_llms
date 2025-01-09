# learn_llms
A personal repo trying to resolve the mess of LLMs / GenAI for my own sake | _Reader beware: this might not make sense! ;-)_ 

Although I have a fair understanding of the universe of AI and atleast the recent trends, putting them to work while also trying to absorb the light speed new things coming out is not easy. Especially with LLMs. I'm trying to find and use/re-use a boring set stack, just like I'd do with Node.JS, to re-implement the same stuff over and over again. However, as I see it, there are too many things. With this repo, I'll try to kind of maintain a running stack with whatever my understanding is.

# High level
1. Training and inference and run locally
    1. Inference
        1. Ollama etc.
        2. oobabooga
        3. Llama.cpp / Llama.cpp python + GGUFs models
        4. HF - might want to use with GGUF models or vice versa
        5. VLLm
        6. langchain (used a lot with RAG stuff as I read)
        7. lighting ai / litgpt
2. Fine tuning
    1. use of embeddings to fine tune an existing model (Mainly used for RAG)- https://ollama.com/blog/embedding-models- Simple fine tuning with OPen AI fine tuning
    2. Where does LORA sit? - Possibly already baked in / Probably OpenAI uses it or some version of it
        1. LoRA is a fine-tune, just on specific weights. It could actually be better at retaining facts than fully training the whole model.
        2. https://arxiv.org/abs/2106.09685 Hugging face literally calls it Parameter-efficient fine-tuning.
        3. https://github.com/huggingface/peft
    3. There is a RAG vs Fine tuning debate, RAG uses vector DBs
        1. https://qdrant.tech/articles/what-is-rag-in-ai/
        2. https://github.com/unslothai/unsloth
3. Agents: - HF - Langchain (somehow mainly used for RAG??) -  Crew AI etc
4. Small models / 1 Bit models
5. WebLLM
6. Web scrapping
    7. https://python.langchain.com/v0.1/docs/use_cases/web_scraping/
    8. https://huggingface.co/blog/luigi12345/ai-scraping-browser-automation
    9. Crawl4AI
    10. https://www.linkedin.com/feed/update/activity:7276642671257927682/?trk=feed_main-feed-card_comment-cta
    11. Stagehand

_There is also a lot of mix and match between Langchain, HF, other libraries etc..... HF seems to have a very extensive use case for everyhing_
https://www.infoworld.com/article/3705035/5-easy-ways-to-run-an-llm-locally.amp.html

# Key resources which I keep seeing
1. Sebastian raska book
2. marcus book
3. https://venturebeat.com/ai/exclusive-browserbase-launches-headless-browser-platform-that-lets-llms-automate-web-tasks/
4. https://www.cnbc.com/2024/06/07/after-chatgpt-and-the-rise-of-chatbots-investors-pour-into-ai-agents.html
5. https://microsaasidea.substack.com/p/micro-saas-ideas-ai-data-videos-knowledgebases?utm_source=post-email-title&publication_id=156669&post_id=145375629&utm_campaign=email-post-title&isFreemail=true&r=1q81q&triedRedirect=true&utm_medium=email

# Todo
1. Read about Phi-3 Simple paper on training, fine tuning, deploying, locally and cloud with  multiple libraries and a use casE?
2. Running through key examples / posts etc. and adding them here
