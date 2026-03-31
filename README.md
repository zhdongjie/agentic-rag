# Agentic-RAG

运行环境：Docker Python3.12.3

## 项目配置

1. **在项目根目录创建虚拟环境**

    ```bash
    python -m venv .venv
    ```

2. **激活虚拟环境**
   ```bash
   # Windows
   .\.venv\Scripts\activate
   
   # macOS / Linux
   source .venv/bin/activate
   ```

## 安装 Poetry

```bash
# 安装 pipx
pip install pipx
pipx ensurepath
```
> 配置生效后请重启 PyChrom
```bash
# 安装 Poetry
pipx install poetry
```

配置 Poetry 在项目目录内创建虚拟环境

```bash
poetry config virtualenvs.in-project true
```

安装项目依赖
```bash
poetry install
```

## 安装向量数据库 Milvus

 ```bash
 wget https://github.com/milvus-io/milvus/releases/download/v2.6.11/milvus-standalone-docker-compose.yml -O docker-compose.yml
 ```

启动 Milvus
 ```bash
 docker compose up -d
 ```

## 安装 Milvus 可视化工具 attu

 ```bash
 docker run -d --name attu -p 3000:3000 --network host zilliz/attu:latest
 ```

安装成功之后访问：http://127.0.0.1:3000
用户名：root
密码：Milvus

## required

将要构建知识库的文档放在 data/docs/markdown 目录下

获取 Dashscope API Key：https://dashscope.aliyun.com

在项目配置文件[config.yaml](config.yaml)中进行配置

## 初始化数据库

 ```bash
poetry run python -m scripts.init_milvus
 ```

## 构建知识库

 ```bash
poetry run python -m script.build_knowledge_base
```

## 启动服务

```bash
poetry run langchain serve
```
API 文档：http://localhost:8000

## Call API

```http request
POST http://127.0.0.1:8000/chat/stream
Content-Type: application/json

{
  "input": {
    "messages": [
      {
        "type": "human",
        "content": "什么是多态？"
      }
    ]
  }
}
```