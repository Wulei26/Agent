# Chroma 数据库 Docker 部署文档

## 一、环境概览

| 项目 | 值 |
|------|-----|
| 宿主机 | Ubuntu 26.04 |
| 数据持久化路径 | `/home/terry/code/learning/Agent/image_similarity_main/test/chroma_db/http_db` |
| 宿主机端口 | `8999` |
| 容器内部端口 | `8000` |
| 镜像 | `ghcr.io/chroma-core/chroma:latest` |
| Compose 文件位置 | `/home/terry/code/learning/Agent/image_similarity_main/test/chroma_db/docker-compose.yml` |

---

## 二、Docker 镜像加速配置

### 2.1 配置 `/etc/docker/daemon.json`

```json
{
    "registry-mirrors": [
        "https://docker.1ms.run",
        "https://docker.xuanyuan.me",
    ]
}
```

> 说明：`registry-mirrors` 只对 Docker Hub (`docker.io`) 生效。拉取 `ghcr.io` 的镜像需要用下面的专用加速域名。

### 2.3 验证镜像源

```bash
docker info | grep -A 5 "Registry Mirrors"
```

---

## 三、拉取 Chroma 镜像

### 3.1 GHCR 专用加速（推荐）

```bash
docker pull ghcr.1ms.run/chroma-core/chroma:latest
```

拉取后重新打标签，方便 compose 使用：

```bash
docker tag ghcr.1ms.run/chroma-core/chroma:latest chromadb/chroma:latest
```


**重载并重启：**

```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
```

---

## 四、Docker Compose 配置

### 4.1 创建数据目录

```bash
mkdir -p /home/terry/code/learning/Agent/image_similarity_main/test/chroma_db/http_db
```

### 4.2 编写 `docker-compose.yml`

```yaml
version: '3.9'

services:
  chroma:
    image: chromadb/chroma:latest
    container_name: chroma-server
    ports:
      - "8999:8000"
    volumes:
      - ./chroma_data:/data
    environment:
      IS_PERSISTENT: "TRUE"
      PERSIST_DIRECTORY: "/data"
      ANONYMIZED_TELEMETRY: "FALSE"
      CHROMA_CORS_ALLOW_ORIGINS: '["http://localhost:8090"]'
    restart: unless-stopped
```

> 端口映射说明：容器内部固定监听 `8000`，宿主机映射为 `8999`，外部访问用 `8999`。

### 4.3 启动服务

```bash
cd /home/terry/code/learning/Agent/image_similarity_main/test/chroma_db
docker compose up -d
```

> 注意：`docker compose up -d` 默认只认当前目录下的 `docker-compose.yml`。也可用 `-f` 指定路径：
> ```bash
> docker compose -f /path/to/docker-compose.yml up -d
> ```

### 4.4 常用管理命令

```bash
docker compose logs -f      # 查看日志
docker compose down         # 停止并移除容器
docker compose restart      # 重启
docker compose ps           # 查看容器状态
```

---

## 五、验证服务

```bash
curl http://localhost:8999/api/v2/heartbeat
```

返回类似以下内容即表示服务正常：

```json
{"nanosecond heartbeat":1791560704892350764}
```

---

## 六、Python 客户端连接

### 6.1 基础连接

```python
import chromadb

client = chromadb.HttpClient(host="localhost", port=8999)

collection = client.get_or_create_collection(name="my_collection")
collection.add(
    ids=["id1", "id2", "id3"],
    documents=[
        "This is a document about pineapple",
        "This is a document about oranges",
        "This is a document about dogs"
    ],
)
```

### 6.2 查看数据

```python
# 列出所有 collection
print(client.list_collections())

# 获取 collection
collection = client.get_collection(name="my_collection")

# 查看总数
print(collection.count())

# 查看全部数据
print(collection.get())

# 语义查询
result = collection.query(query_texts=["fruit"], n_results=2)
print(result['documents'])
```

---

## 七、嵌入模型下载问题

### 7.1 问题说明

Chroma Server 只负责存储和检索向量，不负责生成嵌入。客户端调用 `add()` 时，ChromaDB 会自动下载默认嵌入模型 `all-MiniLM-L6-v2`，下载地址为：

```
https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz
```

国内网络直连该地址可能超时或极慢。

### 7.2 解决方案：手动下载模型

**第一步：下载模型文件**

在浏览器或下载工具中打开上述地址，下载 `onnx.tar.gz`。

**第二步：放到缓存目录**

| 系统 | 路径 |
|------|------|
| Linux / Mac | `~/.cache/chroma/onnx_models/all-MiniLM-L6-v2/` |
| Windows | `C:\Users\用户名\.cache\chroma\onnx_models\all-MiniLM-L6-v2\` |

若 `all-MiniLM-L6-v2` 文件夹不存在需手动创建。放置完成后，ChromaDB 会直接使用本地文件，不再联网下载。

### 7.3 备选：显式传入本地 embedding function

若已有本地 sentence-transformers 模型，可自行包装 embedding function 传入 `get_or_create_collection`，彻底绕过自动下载。

---

## 八、可视化查看工具

### ChromaDB 可视化工具 Chroma UI 使用指南

#### 1. 简介

Chroma UI 是一款基于 Web 的 ChromaDB 可视化管理工具，可以通过浏览器管理 Chroma 向量数据库。

主要功能：

- 查看和管理 Collection（集合）
- 浏览 Documents、Embeddings、Metadata
- 执行向量相似度查询
- 查看、添加和删除数据

项目地址：https://github.com/BlackyDrum/chromadb-ui

#### 2. 启动 ChromaDB

使用 Docker Compose 部署 Chroma Server：

```yaml
services:
  chroma:
    image: chromadb/chroma:latest
    container_name: chroma-server
    ports:
      - "8999:8000"
    volumes:
      - ./chroma_data:/data
    environment:
      IS_PERSISTENT: "TRUE"
      PERSIST_DIRECTORY: "/data"
      ANONYMIZED_TELEMETRY: "FALSE"
      CHROMA_CORS_ALLOW_ORIGINS: '["http://localhost:8090"]'
    restart: unless-stopped
```

启动服务：

```bash
docker compose up -d
```

验证 Chroma Server 是否正常运行：

```bash
curl http://localhost:8999/api/v2/heartbeat
```

**注意：**

- `8999` 是宿主机端口，`8000` 是容器内部端口。
- `CHROMA_CORS_ALLOW_ORIGINS` 用于解决浏览器跨域访问问题。
- CORS 允许来源必须与 Chroma UI 实际访问地址一致。
- 已有数据库应保留原来的持久化目录，避免数据不可见。

#### 3. 安装与启动 Chroma UI

克隆项目：

```bash
git clone https://github.com/BlackyDrum/chromadb-ui.git
cd chromadb-ui
```

安装依赖：

```bash
npm ci
```

启动开发服务器：

```bash
npm run dev
```

浏览器访问：

http://localhost:8090

如果实际启动端口不是 `8090`，需要同步修改 Chroma Server 的 CORS 配置。

#### 4. 连接 ChromaDB

在 Chroma UI 的连接页面填写：

| 参数 | 值 | 说明 |
|---|---|---|
| Server URL | `http://localhost:8999` | Chroma Server 地址 |
| Tenant | `default_tenant` | 默认租户 |
| Database | `default_database` | 默认数据库 |

点击 **Connect to Chroma**，连接成功后即可查看 Collection 及其数据。

##### ChromaDB 数据组织结构

```text
Chroma Server
    |
    └── Tenant (default_tenant)
          |
          └── Database (default_database)
                  |
                  ├── Collection (my_collection)
                  |       ├── id1
                  |       ├── id2
                  |       └── id3
                  |
                  └── Collection (other_collection)
```

- **Tenant**：租户，用于逻辑隔离不同组织或项目的数据。
- **Database**：数据库，属于某个 Tenant。
- **Collection**：向量集合，类似关系型数据库中的数据表。
- **Record**：集合中的记录，包含 ID、Document、Embedding、Metadata 等信息。

#### 5. Python 操作示例

通过 Python 向 Chroma Server 写入数据：

```python
import chromadb

client = chromadb.HttpClient(
    host="localhost",
    port=8999
)

# 创建或获取 Collection
collection = client.get_or_create_collection(
    name="my_collection"
)

# 写入数据
collection.upsert(
    ids=["id1", "id2", "id3"],
    documents=[
        "This is a document about pineapple",
        "This is a document about oranges",
        "This is a document about dogs"
    ]
)

# 查询记录数量
print(collection.count())

# 查看全部记录
print(collection.get())
```

执行后刷新 Chroma UI，即可在 `my_collection` 中查看这三条记录。

#### 6. 常见问题

| 问题 | 原因及解决方法 |
|---|---|
| `Network Error` | 检查 Chroma Server 地址和 CORS 配置 |
| `ERR_CONNECTION_REFUSED` | 检查 Docker 容器和端口映射 |
| 无法查看 Collection | 检查 Tenant 和 Database 是否一致 |
| 浏览器提示 CORS 错误 | 配置 `CHROMA_CORS_ALLOW_ORIGINS` 并重建容器 |
| API 返回 404 | 检查 Chroma UI 与 Chroma Server 的 API 版本兼容性 |

##### 验证 CORS 配置

```bash
curl -i -X OPTIONS \
  http://localhost:8999/api/v2/heartbeat \
  -H "Origin: http://localhost:8090" \
  -H "Access-Control-Request-Method: GET"
```

正常情况下，响应头应包含：

```http
access-control-allow-origin: http://localhost:8090
```

修改 Docker Compose 后重新创建容器：

```bash
docker compose up -d --force-recreate chroma
```

#### 7. 总结

Chroma UI 通过 HTTP API 与 Chroma Server 通信，实现向量数据库的图形化管理。

整体架构：

```text
Python (chromadb.HttpClient)
          |
          v
Chroma Server (Docker)
     localhost:8999
          ^
          |
       HTTP API
          |
Chroma UI (Vite + Web)
     localhost:8090
```

**核心注意事项：**

1. Chroma UI 与 Chroma Server 使用不同端口时，需要正确配置 CORS。
2. Python 和 Chroma UI 使用相同的 Tenant、Database，才能访问相同的 Collection。
3. Docker 部署时需要正确配置持久化目录，避免容器重建后数据丢失。
4. 推荐开发环境使用 `chromadb.HttpClient()` 连接 Chroma Server，方便 Python 和可视化工具共同管理数据库。
---

## 九、目录结构总览

```
/home/terry/code/learning/Agent/image_similarity_main/
├── docker/
│   └── (Docker 相关文件)
└── test/
    └── chroma_db/
        ├── docker-compose.yml          # Compose 配置
        └── http_db/                    # 数据持久化目录
            └── chroma.sqlite3          # SQLite 元数据文件
```

---

## 十、常见问题排查

### 10.1 拉取镜像超时

- 确认 `ghcr.io` 镜像使用 `ghcr.1ms.run` 前缀，而非 `docker.1ms.run`
- 若镜像源不稳定，配置 Docker 守护进程走本地代理（见 3.2）

### 10.2 客户端 `add()` 卡住或超时

- 原因：正在下载默认嵌入模型 `all-MiniLM-L6-v2`
- 解决：手动下载模型文件放到缓存目录（见 7.2）

### 10.3 可视化工具连不上

- 原因：CORS 限制
- 解决：在 compose 中添加 `CHROMA_SERVER_CORS_ALLOW_ORIGINS` 环境变量（见 8.4）

### 10.4 容器启动后数据丢失

- 检查 volumes 挂载路径是否正确
- 确认 `IS_PERSISTENT=TRUE` 已设置