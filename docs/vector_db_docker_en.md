# How to Build Vector DB Docker Services

> A guide for deploying and operating Vector DBs using Docker Compose

---

## 1. Prerequisites

```bash
# Check Docker version (20.10+ recommended)
docker --version

# Check Docker Compose version (2.0+ recommended)
docker compose version
```

---

## 2. Chroma DB

### Docker Compose Configuration

```yaml
# docker-compose.yml
version: '3.8'
services:
  chromadb:
    image: chromadb/chroma:latest
    container_name: chromadb
    volumes:
      - chroma_data:/chroma/chroma
    environment:
      - CHROMA_SERVER_AUTH_CREDENTIALS_PROVIDER=chromadb.auth.token.TokenConfigServerAuthCredentialsProvider
      - CHROMA_SERVER_AUTH_PROVIDER=chromadb.auth.token.TokenAuthServerProvider
      - CHROMA_SERVER_AUTH_CREDENTIALS=your-secret-token  # Optional
      - ALLOW_RESET=TRUE
    ports:
      - "8000:8000"
    restart: unless-stopped

volumes:
  chroma_data:
    driver: local
```

### Starting and Connecting

```bash
# Start the service
docker compose up -d

# Install Python client
pip install chromadb

# Connect from Python
import chromadb
client = chromadb.HttpClient(host="localhost", port=8000)
```

### Key Configuration Options

| Environment Variable | Description | Default |
|----------|------|--------|
| `CHROMA_SERVER_HTTP_PORT` | HTTP port | 8000 |
| `ALLOW_RESET` | Allow data reset | FALSE |
| `ANONYMIZED_TELEMETRY` | Anonymous telemetry | TRUE |

---

## 3. Qdrant

### Docker Compose Configuration

```yaml
# docker-compose.yml
version: '3.8'
services:
  qdrant:
    image: qdrant/qdrant:latest
    container_name: qdrant
    ports:
      - "6333:6333"   # REST API
      - "6334:6334"   # gRPC
    volumes:
      - qdrant_storage:/qdrant/storage
    environment:
      - QDRANT__SERVICE__GRPC_PORT=6334
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/healthz"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  qdrant_storage:
    driver: local
```

### Starting and Connecting

```bash
# Start the service
docker compose up -d

# Install Python client
pip install qdrant-client

# Connect from Python
from qdrant_client import QdrantClient
client = QdrantClient(url="http://localhost:6333")

# Check status
print(client.get_collections())
```

### Web UI Access
- REST API: http://localhost:6333
- Dashboard: http://localhost:6333/dashboard

### Key Configuration Options

| Environment Variable | Description |
|----------|------|
| `QDRANT__SERVICE__HTTP_PORT` | HTTP port |
| `QDRANT__SERVICE__GRPC_PORT` | gRPC port |
| `QDRANT__LOG_LEVEL` | Log level (INFO/DEBUG/ERROR) |
| `QDRANT__STORAGE__STORAGE_PATH` | Storage path |

---

## 4. Weaviate

### Docker Compose Configuration

```yaml
# docker-compose.yml
version: '3.8'
services:
  weaviate:
    image: cr.weaviate.io/semitechnologies/weaviate:1.24.0
    container_name: weaviate
    ports:
      - "8080:8080"
      - "50051:50051"  # gRPC
    volumes:
      - weaviate_data:/var/lib/weaviate
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
      DEFAULT_VECTORIZER_MODULE: 'none'
      ENABLE_MODULES: 'text2vec-openai,generative-openai'
      CLUSTER_HOSTNAME: 'node1'
    restart: unless-stopped

volumes:
  weaviate_data:
    driver: local
```

### Configuration with OpenAI Module

```yaml
version: '3.8'
services:
  weaviate:
    image: cr.weaviate.io/semitechnologies/weaviate:1.24.0
    environment:
      OPENAI_APIKEY: ${OPENAI_API_KEY}
      DEFAULT_VECTORIZER_MODULE: 'text2vec-openai'
      ENABLE_MODULES: 'text2vec-openai,generative-openai'
    # ... rest of configuration
```

### Starting and Connecting

```bash
# Start the service
docker compose up -d

# Install Python client
pip install weaviate-client

# Connect from Python (v4 client)
import weaviate
client = weaviate.connect_to_local(host="localhost", port=8080)
```

---

## 5. Milvus

### Docker Compose Configuration (Standalone Mode)

Milvus depends on etcd (metadata) and MinIO (object storage).

```yaml
# docker-compose.yml
version: '3.8'
services:
  etcd:
    container_name: milvus-etcd
    image: quay.io/coreos/etcd:v3.5.5
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
      - ETCD_SNAPSHOT_COUNT=50000
    volumes:
      - etcd_data:/etcd
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd
    healthcheck:
      test: ["CMD", "etcdctl", "endpoint", "health"]
      interval: 30s
      timeout: 20s
      retries: 3

  minio:
    container_name: milvus-minio
    image: minio/minio:RELEASE.2023-03-13T19-46-17Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    volumes:
      - minio_data:/minio_data
    command: minio server /minio_data --console-address ":9001"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  milvus:
    container_name: milvus-standalone
    image: milvusdb/milvus:v2.4.0
    command: ["milvus", "run", "standalone"]
    security_opt:
      - seccomp:unconfined
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    volumes:
      - milvus_data:/var/lib/milvus
    ports:
      - "19530:19530"  # gRPC
      - "9091:9091"    # HTTP
    depends_on:
      - etcd
      - minio
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9091/healthz"]
      interval: 30s
      timeout: 20s
      retries: 3

volumes:
  etcd_data:
  minio_data:
  milvus_data:
```

### Starting and Connecting

```bash
# Start the service
docker compose up -d

# Install Python client
pip install pymilvus

# Connect from Python
from pymilvus import connections
connections.connect(host="localhost", port="19530")
```

---

## 6. Integrated Docker Compose (Selective Use)

Integrated configuration for selectively using Vector DBs in RAG services:

```yaml
# docker-compose.yml
version: '3.8'

services:
  # ===== Vector DB Options (activate only one) =====

  # Option 1: Chroma (recommended for development/prototype)
  chromadb:
    image: chromadb/chroma:latest
    container_name: chromadb
    volumes:
      - chroma_data:/chroma/chroma
    ports:
      - "8000:8000"
    restart: unless-stopped
    profiles: ["chroma"]

  # Option 2: Qdrant (recommended for production)
  qdrant:
    image: qdrant/qdrant:latest
    container_name: qdrant
    volumes:
      - qdrant_storage:/qdrant/storage
    ports:
      - "6333:6333"
      - "6334:6334"
    restart: unless-stopped
    profiles: ["qdrant"]

  # Option 3: Weaviate (recommended for hybrid search)
  weaviate:
    image: cr.weaviate.io/semitechnologies/weaviate:1.24.0
    container_name: weaviate
    volumes:
      - weaviate_data:/var/lib/weaviate
    ports:
      - "8080:8080"
    environment:
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
      DEFAULT_VECTORIZER_MODULE: 'none'
    restart: unless-stopped
    profiles: ["weaviate"]

  # ===== RAG Service =====
  app:
    build: .
    container_name: docschat-app
    ports:
      - "8501:8501"  # Streamlit
    environment:
      - VECTOR_DB_TYPE=${VECTOR_DB_TYPE:-chroma}
      - CHROMA_HOST=chromadb
      - QDRANT_HOST=qdrant
      - WEAVIATE_HOST=weaviate
    depends_on:
      - chromadb
      - qdrant
      - weaviate
    volumes:
      - ./data:/app/data
    restart: unless-stopped

volumes:
  chroma_data:
  qdrant_storage:
  weaviate_data:
```

### Profile-based Execution

```bash
# Run only Chroma
docker compose --profile chroma up -d

# Run only Qdrant
docker compose --profile qdrant up -d

# Run only Weaviate
docker compose --profile weaviate up -d
```

---

## 7. Persistence and Backup

### Volume Management

```bash
# List volumes
docker volume ls

# Volume details
docker volume inspect chroma_data

# Backup volume
docker run --rm \
  -v chroma_data:/source \
  -v $(pwd)/backup:/backup \
  alpine tar czf /backup/chroma_backup.tar.gz -C /source .

# Restore volume
docker run --rm \
  -v chroma_data:/target \
  -v $(pwd)/backup:/backup \
  alpine tar xzf /backup/chroma_backup.tar.gz -C /target
```

---

## 8. Monitoring

### Status Check

```bash
# Chroma status
curl http://localhost:8000/api/v1/heartbeat

# Qdrant status
curl http://localhost:6333/healthz

# Weaviate status
curl http://localhost:8080/v1/.well-known/ready

# Milvus status
curl http://localhost:9091/healthz
```

### Docker Resource Monitoring

```bash
# Running container status
docker compose ps

# Resource usage (CPU, memory)
docker stats chromadb

# View logs
docker compose logs -f chromadb
```

---

## 9. Recommendations

| Scenario | Recommended DB | Reason |
|---------|---------|------|
| Local development/testing | Chroma | Simple setup, quick start |
| Small-scale production | Qdrant | Performance/stability balance |
| Hybrid search required | Weaviate | Optimized for complex search |
| Large-scale enterprise | Milvus | Handles billions of vectors |

---

## 10. References

- [Chroma Docker Hub](https://hub.docker.com/r/chromadb/chroma)
- [Qdrant Docker Docs](https://qdrant.tech/documentation/guides/installation/)
- [Weaviate Docker Setup](https://weaviate.io/developers/weaviate/installation/docker-compose)
- [Milvus Docker Docs](https://milvus.io/docs/install_standalone-docker.md)
