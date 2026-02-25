# Vector DB Docker 서비스 구축 방법

> Docker Compose를 활용한 Vector DB 배포 및 운영 가이드

---

## 1. 사전 준비

```bash
# Docker 버전 확인 (20.10+ 권장)
docker --version

# Docker Compose 버전 확인 (2.0+ 권장)
docker compose version
```

---

## 2. Chroma DB

### Docker Compose 설정

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
      - CHROMA_SERVER_AUTH_CREDENTIALS=your-secret-token  # 선택사항
      - ALLOW_RESET=TRUE
    ports:
      - "8000:8000"
    restart: unless-stopped

volumes:
  chroma_data:
    driver: local
```

### 실행 및 연결

```bash
# 서비스 시작
docker compose up -d

# Python 클라이언트 설치
pip install chromadb

# Python에서 연결
import chromadb
client = chromadb.HttpClient(host="localhost", port=8000)
```

### 주요 설정 옵션

| 환경변수 | 설명 | 기본값 |
|----------|------|--------|
| `CHROMA_SERVER_HTTP_PORT` | HTTP 포트 | 8000 |
| `ALLOW_RESET` | 데이터 초기화 허용 | FALSE |
| `ANONYMIZED_TELEMETRY` | 익명 텔레메트리 | TRUE |

---

## 3. Qdrant

### Docker Compose 설정

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

### 실행 및 연결

```bash
# 서비스 시작
docker compose up -d

# Python 클라이언트 설치
pip install qdrant-client

# Python에서 연결
from qdrant_client import QdrantClient
client = QdrantClient(url="http://localhost:6333")

# 상태 확인
print(client.get_collections())
```

### 웹 UI 접근
- REST API: http://localhost:6333
- 대시보드: http://localhost:6333/dashboard

### 주요 설정 옵션

| 환경변수 | 설명 |
|----------|------|
| `QDRANT__SERVICE__HTTP_PORT` | HTTP 포트 |
| `QDRANT__SERVICE__GRPC_PORT` | gRPC 포트 |
| `QDRANT__LOG_LEVEL` | 로그 레벨 (INFO/DEBUG/ERROR) |
| `QDRANT__STORAGE__STORAGE_PATH` | 스토리지 경로 |

---

## 4. Weaviate

### Docker Compose 설정

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

### OpenAI 모듈 포함 설정

```yaml
version: '3.8'
services:
  weaviate:
    image: cr.weaviate.io/semitechnologies/weaviate:1.24.0
    environment:
      OPENAI_APIKEY: ${OPENAI_API_KEY}
      DEFAULT_VECTORIZER_MODULE: 'text2vec-openai'
      ENABLE_MODULES: 'text2vec-openai,generative-openai'
    # ... 나머지 설정
```

### 실행 및 연결

```bash
# 서비스 시작
docker compose up -d

# Python 클라이언트 설치
pip install weaviate-client

# Python에서 연결 (v4 클라이언트)
import weaviate
client = weaviate.connect_to_local(host="localhost", port=8080)
```

---

## 5. Milvus

### Docker Compose 설정 (Standalone 모드)

Milvus는 etcd(메타데이터)와 MinIO(오브젝트 스토리지)에 의존합니다.

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

### 실행 및 연결

```bash
# 서비스 시작
docker compose up -d

# Python 클라이언트 설치
pip install pymilvus

# Python에서 연결
from pymilvus import connections
connections.connect(host="localhost", port="19530")
```

---

## 6. 통합 Docker Compose (선택적 사용)

RAG 서비스에서 Vector DB를 선택적으로 사용하기 위한 통합 구성:

```yaml
# docker-compose.yml
version: '3.8'

services:
  # ===== Vector DB 옵션 (하나만 활성화) =====
  
  # 옵션 1: Chroma (개발/프로토타입 권장)
  chromadb:
    image: chromadb/chroma:latest
    container_name: chromadb
    volumes:
      - chroma_data:/chroma/chroma
    ports:
      - "8000:8000"
    restart: unless-stopped
    profiles: ["chroma"]

  # 옵션 2: Qdrant (운영 환경 권장)
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

  # 옵션 3: Weaviate (하이브리드 검색 권장)
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

  # ===== RAG 서비스 =====
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

### Profile 기반 실행

```bash
# Chroma만 실행
docker compose --profile chroma up -d

# Qdrant만 실행
docker compose --profile qdrant up -d

# Weaviate만 실행
docker compose --profile weaviate up -d
```

---

## 7. 영속성 및 백업

### 볼륨 관리

```bash
# 볼륨 목록 확인
docker volume ls

# 볼륨 상세 정보
docker volume inspect chroma_data

# 볼륨 백업
docker run --rm \
  -v chroma_data:/source \
  -v $(pwd)/backup:/backup \
  alpine tar czf /backup/chroma_backup.tar.gz -C /source .

# 볼륨 복원
docker run --rm \
  -v chroma_data:/target \
  -v $(pwd)/backup:/backup \
  alpine tar xzf /backup/chroma_backup.tar.gz -C /target
```

---

## 8. 모니터링

### 상태 확인

```bash
# Chroma 상태
curl http://localhost:8000/api/v1/heartbeat

# Qdrant 상태
curl http://localhost:6333/healthz

# Weaviate 상태
curl http://localhost:8080/v1/.well-known/ready

# Milvus 상태
curl http://localhost:9091/healthz
```

### Docker 리소스 모니터링

```bash
# 실행 중인 컨테이너 상태
docker compose ps

# 리소스 사용량 (CPU, 메모리)
docker stats chromadb

# 로그 확인
docker compose logs -f chromadb
```

---

## 9. 권장 사항

| 시나리오 | 권장 DB | 이유 |
|---------|---------|------|
| 로컬 개발/테스트 | Chroma | 간단한 설정, 빠른 시작 |
| 소규모 프로덕션 | Qdrant | 성능/안정성 균형 |
| 하이브리드 검색 필요 | Weaviate | 복합 검색 최적화 |
| 대규모 엔터프라이즈 | Milvus | 수십억 벡터 처리 |

---

## 10. 참고 자료

- [Chroma Docker Hub](https://hub.docker.com/r/chromadb/chroma)
- [Qdrant Docker Docs](https://qdrant.tech/documentation/guides/installation/)
- [Weaviate Docker Setup](https://weaviate.io/developers/weaviate/installation/docker-compose)
- [Milvus Docker Docs](https://milvus.io/docs/install_standalone-docker.md)
