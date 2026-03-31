# Agentic-RAG

运行环境：Docker Python3.12.3

1. 安装向量数据库 Milvus

    ```bash
    wget https://github.com/milvus-io/milvus/releases/download/v2.6.11/milvus-standalone-docker-compose.yml -O docker-compose.yml
    ```

   启动 Milvus
    ```bash
    docker compose up -d
    ```

2. 安装 Milvus 可视化工具 attu

    ```bash
    docker run -d --name attu -p 3000:3000 --network host zilliz/attu:latest
    ```

   安装成功之后访问：http://127.0.0.1:3000
   用户名：root
   密码：Milvus

3. 将要构建知识库的文档放在 data/docs/markdown 目录下

4. 初始化数据库
    ```bash
    script python init_milvus.py
    ```

5. 构建知识库
    ```bash
   script python build_knowledge_base.py
   ```

6. 启动服务
   在 app/main.py 右击运行；
   API 文档：http://localhost:5000

7. Call API
   ```http request
   POST http://127.0.0.1:5000/chat/stream
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