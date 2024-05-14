# Chatbot Glossary

|Concept   	|Definition   	|
|---	|---	|
| LLM   	| Large Language model, notable for its ability to generate language. Synonym of Chatbot.   	|
| Model size / Number of Parameters    | The model size is the number of parameters in the LLM. The more parameters a model has, the more complex it is and the more data it can process. However, larger models are also more computationally expensive to train and deploy.|
| Prompt  	| The question asked to a chatbot.  	|
| Temperature  	| Creativity of the answers of the model. Higher values will generate more diverse outputs. Lowers values will deliver more conservative and deterministic results.  	|
| Halucination  	| factually incorrect, nonsensical, or disconnected answer from the input prompt  	|
| RAG   	| Retrieval-augmented generation, a concept allowing a model to generate answers retrieved from custom sources of knowledge (eg: private documentation, FAQ etc...) 	|


# Prerequisites

-   Docker
```
sudo curl https://get.docker.com/ | sh 
```

- Cuda ???

- Nvidia container toolkit
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installation


- Configure Docker to use Nvidia driver
```sh
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

# Backend
[ollama](https://ollama.com/download/linux) is a tool to deploy pretrained opensource AI models locally.

It can be easilly installed with few commands :
```sh
curl -fsSL https://ollama.com/install.sh | sh
ollama run llama3

curl http://localhost:11434/api/generate -d '{"model": "llama3", "prompt": "Why is the sky blue?"}'
```

# Web client
[OpenwebUI](https://github.com/open-webui/open-webui) is a web interface which uses an ollama backend.
It can be easilly installed with docker :
```sh
docker run -d -p 3000:8080 --gpus all --add-host=host.docker.internal:host-gateway -v open-webui:/app/backend/data --name open-webui ghcr.io/open-webui/open-webui:cuda
```

# Docker compose command
Copy the `docker-compose.yaml`  file 
Run :
```sh
docker compose -f docker-compose.yaml up -d
```
It will deploy two containers ollama and openwebUI.

Access the web UI using http://YOUR-IP:3000


# OpenWebUI configuration
1. Create a new account
2. Select the llm model
3. Add data
4. Ask question


# Shut down the service
```sh
docker compose -f docker-compose.yaml down
```