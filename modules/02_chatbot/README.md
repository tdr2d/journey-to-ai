# Chatbot Glossary

|Concept   	|Definition   	|
|---	|---	|
| LLM   	| Large Language model, notable for its ability to generate language. Synonym of Chatbot.   	|
| Model size / Number of Parameters    | The model size is the number of parameters in the LLM. The more parameters a model has, the more complex it is and the more data it can process. However, larger models are also more computationally expensive to train and deploy.|
| Prompt  	| The question asked to a chatbot.  	|
| Training   	| Learning (determining) good values for all the weights and the bias from labeled examples.. 	|
| Inference   	| Using an AI trained model with input data (prompt) to obtain an answer. 	|
| Temperature  	| Creativity of the answers of the model. Higher values will generate more diverse outputs. Lowers values will deliver more conservative and deterministic results.  	|
| Halucination  	| factually incorrect, nonsensical, or disconnected answer from the input prompt  	|
| RAG   	| Retrieval-augmented generation, a concept allowing a model to generate answers retrieved from custom sources of knowledge (eg: private documentation, FAQ etc...) 	|


# Prerequisites
- A Cuda and docker compatible OS. You can use the NVIDIA GPU Cloud (NGC) OS which already includes CUDA + Docker
- Otherwise you can to install yourself :
    - [CUDA](https://developer.nvidia.com/cuda-downloads)
    - [Docker](https://docs.docker.com/engine/install/)
    - [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- Testing cuda + nvidia docker driver
```sh
docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
# It should show the following text message to retrieve th GPU consumption
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.15              Driver Version: 550.54.15      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla V100-PCIE-16GB           Off |   00000000:00:06.0 Off |                    0 |
| N/A   39C    P0             28W /  250W |       4MiB /  16384MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
+-----------------------------------------------------------------------------------------+
```

# Tools
[Ollama](https://ollama.com/download/linux) is a tool to deploy pretrained opensource AI models locally.

[OpenwebUI](https://github.com/open-webui/open-webui) is a web interface which uses an ollama backend.


# Install commands
To easilly install ollama + openwebUI in one command, we will use docker.<br>
This command will deploy ollama and openWebUI locally :
```sh
docker run -d -p 3000:8080 --gpus=all -v ollama:/root/.ollama -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:ollama
```
It will deploy two openwebUI and ollama in a single container.

It will expose the openWebUI on the url: http://PUBLIC-INSTANCE-IP:3000


# OpenWebUI configuration
1. After you access the OpenWebUI page for the first time, you need to Create a new account
2. Select the llm model in the top dropdown input menu (select and download llama3). The list of Opensource models are displayed here: https://ollama.com/library
3. Add your custom data using the plus logo inside the bottom text input area.
4. Ask question selecting your data.


# Shut down the service
To delete the container and the data run the following command :
```sh
docker rm open-webui --volumes
```