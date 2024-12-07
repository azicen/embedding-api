# embedding-bge-m3-api

使用BGE-M3模拟openai文本向量转换接口 `/v1/embeddings`

> BGE-M3是多语言语义向量模型，支持最高8192序列长度，输出1024长度的向量。BGE-M3已向社区全面开源并支持免费商用许可。

## 使用

### 快速启动

```sh
pip install -r requirements.txt
# 需要使用cuda请修改为 requirements.cuda.txt

python main.py
```

### 通过docker部署

```sh
docker run \
  --name embedding-bge-m3-api \
  --restart=always \
  -e TZ="Asia/Shanghai" \
  -p "8000:8000" \
  #--gpus all \
  -d \
  azicen/embedding-bge-m3-api:latest
  #azicen/embedding-bge-m3-api:latest-cuda
```

### 通过docker-compose部署

```yaml
version: '3.8'

services:
  embedding-bge-m3-api:
    image: "azicen/embedding-bge-m3-api:latest"
    # image: "azicen/embedding-bge-m3-api:latest-cuda"
    container_name: embedding-bge-m3-api
    environment:
      TZ: "Asia/Shanghai"
      ROUTE_PREFIX: ""
      API_KEY: ""
      EMBEDDING_MODEL: "BAAI/bge-m3-unsupervised"
    ports:
      - "8000:8000"
    #deploy:
    #  resources:
    #    reservations:
    #      devices:
    #        - capabilities: ["gpu"]
```

## HTTP-API

参考[openai文档](https://platform.openai.com/docs/api-reference/embeddings/create)

### Default

```sh
curl --request POST \
  --url http://localhost:8000/v1/embeddings \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "The food was delicious and the waiter...",
    "model": "any-value",
    "encoding_format": "float"
  }'
```

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [
        0.0023064255,
        -0.009327292,
        ....
        -0.0028842222,
      ],
      "index": 0
    }
  ],
  "model": "text-embedding-ada-002",
  "usage": {
    "prompt_tokens": 8,
    "total_tokens": 8
  }
}
```

## 环境变量

| 变量名               | 描述                              | 默认值 |
| -------------------- | --------------------------------- | ------ |
| ROUTE_PREFIX         | HTTP接口前缀                      | nil    |
| DEVICE               | 加载模型的设备, 选其一: cpu或cuda | cuda   |
| API_KEY              | HTTP接口授权API-KEY               | nil    |
| CUDA_LAUNCH_BLOCKING | CUDA启动阻塞                      | 1      |
