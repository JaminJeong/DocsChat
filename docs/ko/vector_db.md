# Vector Database 종류별 분석

> RAG(Retrieval-Augmented Generation) 기반 Chat 서비스를 위한 Vector DB 비교 분석

---

## 1. Vector Database란?

Vector Database는 고차원 벡터(임베딩)를 저장하고, 유사도 기반 검색(ANN: Approximate Nearest Neighbor)을 효율적으로 수행하는 데이터베이스입니다.

### 핵심 개념
- **Embedding**: 텍스트, 이미지 등을 고차원 수치 벡터로 변환한 것
- **ANN Search**: 주어진 쿼리 벡터와 유사한 벡터를 빠르게 검색
- **Index**: 검색 최적화를 위한 자료구조 (HNSW, IVF, PQ 등)

---

## 2. 주요 Vector DB 비교

### 2.1 Chroma

| 항목 | 내용 |
|------|------|
| **타입** | 오픈소스, Self-hosted / Cloud |
| **언어** | Python (기본), JavaScript |
| **라이선스** | Apache 2.0 |
| **GitHub** | [chroma-core/chroma](https://github.com/chroma-core/chroma) |

**특징**
- AI Native 임베딩 데이터베이스로 개발자 친화적 설계
- 설치 및 사용이 매우 간단 (`pip install chromadb`)
- 인메모리 모드와 영구 저장 모드 모두 지원
- LangChain, LlamaIndex와 완벽한 통합
- 2024년 멀티모달 임베딩(OpenCLIP) 지원 추가

**장점**
- 빠른 프로토타이핑에 최적
- 별도 서버 없이 Python 라이브러리로 시작 가능
- 직관적인 API

**단점**
- 대규모(수십억 벡터) 환경에서는 성능 한계
- 엔터프라이즈 기능 부족

**적합한 용도**: 프로토타입, 소규모 RAG 프로젝트, 개발/테스트

```python
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.create_collection("documents")
collection.add(
    embeddings=[[1.0, 2.0, 3.0]],
    documents=["Sample document"],
    ids=["doc1"]
)
results = collection.query(query_embeddings=[[1.0, 2.0, 3.0]], n_results=5)
```

---

### 2.2 Qdrant

| 항목 | 내용 |
|------|------|
| **타입** | 오픈소스, Self-hosted / Cloud |
| **언어** | Rust (코어), Python/Go/JS 클라이언트 |
| **라이선스** | Apache 2.0 |
| **GitHub** | [qdrant/qdrant](https://github.com/qdrant/qdrant) |

**특징**
- Rust로 개발된 고성능 벡터 검색 엔진
- 풍부한 JSON 기반 페이로드 필터링
- 실시간 임베딩 검색 최적화
- 양자화(Quantization) 옵션으로 메모리 최적화 (2024년 확장)
- REST API, gRPC 지원

**장점**
- 뛰어난 성능과 안정성
- 고급 메타데이터 필터링
- Docker로 간단히 배포 가능
- 유연한 배포 옵션 (로컬/클라우드)

**단점**
- Chroma에 비해 설정이 약간 복잡
- 상대적으로 최근에 등장한 프로젝트

**적합한 용도**: 고성능 RAG, 복잡한 메타데이터 필터링이 필요한 서비스

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient(url="http://localhost:6333")
client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
)
```

---

### 2.3 Weaviate

| 항목 | 내용 |
|------|------|
| **타입** | 오픈소스, Self-hosted / Cloud (Weaviate Cloud) |
| **언어** | Go (코어), Python/JS/Go 클라이언트 |
| **라이선스** | BSD 3-Clause |
| **GitHub** | [weaviate/weaviate](https://github.com/weaviate/weaviate) |

**특징**
- 그래프 기반 벡터 데이터베이스
- 하이브리드 검색(벡터 + 키워드 + 메타데이터 필터) 강점
- GraphQL API 지원
- 멀티테넌시, 멀티모달 지원
- 모듈 시스템으로 확장 가능 (text2vec, img2vec 등)

**장점**
- 하이브리드 검색에서 탁월한 성능
- 지식 그래프 구현 가능
- 다양한 임베딩 모델 모듈 지원

**단점**
- 설정 및 스키마 정의가 복잡
- 메모리 사용량이 높은 편

**적합한 용도**: 복잡한 데이터 관계, 하이브리드 검색, 엔터프라이즈 RAG

```python
import weaviate

client = weaviate.Client("http://localhost:8080")
client.schema.create_class({
    "class": "Document",
    "vectorizer": "text2vec-openai",
    "properties": [{"name": "content", "dataType": ["text"]}]
})
```

---

### 2.4 Milvus

| 항목 | 내용 |
|------|------|
| **타입** | 오픈소스, Self-hosted / Zilliz Cloud |
| **언어** | Go/C++ (코어), Python/Java/Go 클라이언트 |
| **라이선스** | Apache 2.0 |
| **GitHub** | [milvus-io/milvus](https://github.com/milvus-io/milvus) |

**특징**
- 클라우드 네이티브 분산 아키텍처 (컴퓨팅/스토리지 분리)
- 10가지 이상의 인덱스 타입 지원 (GPU 인덱싱 포함)
- 수십억 ~ 수조 개 벡터 처리 가능
- 스칼라 필터 검색 고도화

**장점**
- 초대규모 엔터프라이즈 환경에 최적
- GPU 가속 지원
- 높은 처리량과 낮은 지연시간

**단점**
- 설정 및 운영이 복잡 (etcd, MinIO 의존성)
- 소규모 프로젝트에는 오버스펙
- 리소스 요구사항이 높음

**적합한 용도**: 대규모 엔터프라이즈 RAG, 수십억 벡터 처리

```python
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection

connections.connect(host="localhost", port="19530")
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536)
]
schema = CollectionSchema(fields)
collection = Collection("documents", schema)
```

---

### 2.5 Pinecone

| 항목 | 내용 |
|------|------|
| **타입** | 완전 관리형 클라우드 서비스 |
| **언어** | Python/JS/Go 클라이언트 |
| **라이선스** | 상용 (무료 티어 존재) |
| **웹사이트** | [pinecone.io](https://www.pinecone.io) |

**특징**
- 완전 관리형(Serverless) 벡터 데이터베이스
- 스케일링 복잡성 완전 추상화
- 고처리량, 저지연 벡터 검색
- LangChain, LlamaIndex와 긴밀한 통합

**장점**
- 인프라 관리 불필요
- 기업 수준의 안정성과 신뢰성
- 간단한 API

**단점**
- 유료 서비스 (비용 발생)
- 인터넷 연결 필수 (로컬 배포 불가)
- 데이터 외부 전송 필요

**적합한 용도**: 운영 관리를 최소화하고 싶은 프로덕션 환경

---

### 2.6 FAISS (Facebook AI Similarity Search)

| 항목 | 내용 |
|------|------|
| **타입** | 오픈소스 라이브러리 |
| **언어** | C++ (코어), Python 래퍼 |
| **라이선스** | MIT |
| **GitHub** | [facebookresearch/faiss](https://github.com/facebookresearch/faiss) |

**특징**
- 완전한 데이터베이스가 아닌 유사도 검색 라이브러리
- 인메모리 기반으로 가장 빠른 ANN 검색 성능
- GPU 지원
- LangChain에서 로컬 벡터스토어로 자주 사용

**장점**
- 압도적인 검색 속도 (인메모리)
- GPU 가속 지원
- 연구/실험에 최적

**단점**
- 독립적인 데이터베이스 기능 없음 (API 서버, 영구 저장 등)
- 수동 영속성 관리 필요
- 확장성 제한

**적합한 용도**: 연구, 고성능 인메모리 검색, 커스텀 솔루션

```python
import faiss
import numpy as np

d = 1536  # dimension
index = faiss.IndexFlatL2(d)
vectors = np.random.random((100, d)).astype('float32')
index.add(vectors)
D, I = index.search(np.random.random((1, d)).astype('float32'), k=5)
```

---

## 3. 비교 요약표

| 항목 | Chroma | Qdrant | Weaviate | Milvus | Pinecone | FAISS |
|------|--------|--------|----------|--------|----------|-------|
| **타입** | 오픈소스 | 오픈소스 | 오픈소스 | 오픈소스 | 상용 관리형 | 라이브러리 |
| **Docker 지원** | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| **Self-hosted** | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| **확장성** | 중간 | 높음 | 높음 | 매우 높음 | 매우 높음 | 낮음 |
| **설정 난이도** | 매우 쉬움 | 쉬움 | 보통 | 어려움 | 매우 쉬움 | 쉬움 |
| **하이브리드 검색** | 제한적 | ✅ | ✅ (탁월) | ✅ | ✅ | ❌ |
| **멀티모달** | ✅ (2024) | ✅ | ✅ | ✅ | ✅ | 제한적 |
| **LangChain 통합** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **비용** | 무료 | 무료/유료 | 무료/유료 | 무료/유료 | 유료 | 무료 |
| **적합 규모** | 소~중 | 중~대 | 중~대 | 대~초대 | 중~대 | 소~중 |

---

## 4. RAG 서비스를 위한 권장사항

### 🚀 빠른 시작 / 프로토타입
→ **Chroma** 또는 **FAISS**
- 간단한 설치, 코드 몇 줄로 시작
- LangChain과 즉시 연동 가능

### ⚡ 성능 중심 / 운영 환경
→ **Qdrant**
- Docker로 쉽게 배포, 고성능, 풍부한 필터링
- DocsChat 프로젝트에 **가장 추천**

### 🔍 하이브리드 검색 / 복잡한 쿼리
→ **Weaviate**
- 벡터 + 키워드 + 메타데이터 복합 검색

### 🏢 대규모 엔터프라이즈
→ **Milvus** 또는 **Pinecone**
- 수십억 벡터, 높은 처리량 요구 환경

---

## 5. 참고 자료

- [Chroma 공식 문서](https://docs.trychroma.com)
- [Qdrant 공식 문서](https://qdrant.tech/documentation)
- [Weaviate 공식 문서](https://weaviate.io/developers/weaviate)
- [Milvus 공식 문서](https://milvus.io/docs)
- [FAISS GitHub](https://github.com/facebookresearch/faiss)
- [LangChain Vector Stores](https://python.langchain.com/docs/integrations/vectorstores/)
