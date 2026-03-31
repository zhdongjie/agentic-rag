# 🚀 Agentic-RAG 项目指南

## 🧱 一、运行环境

- Python: 3.12.3  
- 推荐方式：Docker + Poetry  
- 支持系统：Windows / macOS / Linux  

---

## 📦 二、项目初始化

### ✅ 推荐：使用 Poetry（不要和 venv 混用）

```bash
poetry install
```

### ⚠️ 可选：手动 venv（不推荐）

```bash
python -m venv .venv
```

激活环境：

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

---

## 🧰 三、安装 Poetry

```bash
pip install pipx
pipx ensurepath
```

> ⚠️ 执行后请重启终端 / IDE

安装 Poetry：

```bash
pipx install poetry
```

设置虚拟环境在项目目录：

```bash
poetry config virtualenvs.in-project true
```

安装依赖：

```bash
poetry install
```

---

## 🧠 四、安装 Milvus（向量数据库）

### 1️⃣ 下载配置文件

```bash
wget https://github.com/milvus-io/milvus/releases/download/v2.6.11/milvus-standalone-docker-compose.yml -O docker-compose.yml
```

### 2️⃣ 启动服务

```bash
docker compose up -d
```

### 3️⃣ 检查运行状态

```bash
docker ps
```

应包含：
- milvus
- etcd
- minio

---

## 📊 五、安装 Attu（Milvus 可视化工具）

```bash
docker run -d \
  --name attu \
  -p 3000:3000 \
  --network host \
  zilliz/attu:latest
```

访问：

http://127.0.0.1:3000

登录信息：

用户名：root  
密码：Milvus  

### ⚠️ 注意

- macOS 可能不支持 `--network host`
- 可改为 bridge 模式
- 连接地址填写：`localhost:19530`

---

## 📁 六、数据准备

将文档放入目录：

```
data/docs/markdown/
```

### 建议：

- 使用 `.md` 格式  
- 单文件 < 2MB  
- 避免超长段落（影响 embedding）  

---

## 🔑 七、配置 API Key

获取地址：  
https://dashscope.aliyun.com  

编辑配置文件：

```yaml
# config.yaml
api-key: "your_api_key_here"
```

### ⚠️ 建议

- 加入 `.gitignore`  
- 不要提交到仓库  

---

## 🗄️ 八、初始化数据库

```bash
poetry run python -m scripts.init_milvus
```

---

## 📚 九、构建知识库

```bash
poetry run python -m scripts.build_knowledge_base
```

---

## 🚀 十、启动服务

```bash
poetry run langchain serve
```

API 文档：  
http://localhost:8000  

---

## 🔌 十一、调用接口

### 请求

```http
POST http://127.0.0.1:8000/chat/stream
Content-Type: application/json
```

### Body

```json
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
