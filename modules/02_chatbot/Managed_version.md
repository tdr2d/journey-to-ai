# Step 1: Generate API Access in OVHCLoud AI-ENDPOINT
https://help.ovhcloud.com/csm/fr-public-cloud-ai-endpoints-getting-started?id=kb_article_view&sysparm_article=KB0065411


```bash
export OVH_AI_ENDPOINTS_API_KEY=eyJhbGciOiJFZERTQSIsImtpZCI6IjgzMkFGNUE5ODg3MzFCMDNGM0EzMTRFMDJFRUJFRjBGNDE5MUY0Q0YiLCJraW5kIjoicGF0Iiwkcwuy-KOO49AklW0_QH93gV_LAQ
```

# Step 2: Create a linux instance with docker
https://help.ovhcloud.com/csm/en-ie-public-cloud-compute-getting-started?id=kb_article_view&sysparm_article=KB0051014


# Step 3: Deploy OpenWebUI
```bash
docker run -d --name open-webui -p 3000:8080 \
    -e OPENAI_API_KEY=$OVH_AI_ENDPOINTS_API_KEY \
    -e OPENAI_API_BASE_URL="https://oai.endpoints.kepler.ai.cloud.ovh.net/v1" \
    -e RAG_OPENAI_API_BASE_URL="https://oai.endpoints.kepler.ai.cloud.ovh.net/v1/embeddings" \
    -e RAG_EMBEDDING_ENGINE="openai" \
    -e RAG_EMBEDDING_MODEL="bge-multilingual-gemma2"  \
    -v open-webui-data:/app/backend/data ghcr.io/open-webui/open-webui:main
```



docker run --rm --name open-webui -p 3000:8080 -e OPENAI_API_KEY=$OVH_AI_ENDPOINTS_API_KEY -e OPENAI_API_BASE_URL="https://oai.endpoints.kepler.ai.cloud.ovh.net/v1"  -e RAG_OPENAI_API_BASE_URL="https://oai.endpoints.kepler.ai.cloud.ovh.net/v1/embeddings" -e RAG_EMBEDDING_ENGINE="openai" -e RAG_EMBEDDING_MODEL="bge-multilingual-gemma2"  -v open-webui-data:/app/backend/data   ghcr.io/open-webui/open-webui:main  


### Recommended completion model:
* Qwen2.5-VL-72B-Instruct

### Recommended Embeddings model :
* bge-multilingual-gemma2

# RAG FAQ
### What are embeddings model ?
This model converts the data from text/image/sound to a vector a machine can better use to perform computations.
The model is used only to perform RAG, when you need to analyse additional documents which were not used when training the completion model.

### Why do you need a Vector Database ?
The vector Database is necesary for RAG. The knowledge documents are translated into vector via the Embedding model and then stored as vectors.
The vectors are searched when prompting to feed the LLM model with context.
