# 시스템 아키텍처

## 1. 구성 요소
- Ingestion Layer: PDF 로딩, 청크 분할, 메타데이터 부착
- Index Layer: Chroma 벡터 DB 임베딩 저장
- Retrieval Layer: BM25 + Vector Search 하이브리드 검색
- Generation Layer: LangChain + LLM 응답 생성
- Guardrail Layer: 출처 강제 표기(파일명/페이지), 컨텍스트 외 추론 금지

## 2. 데이터 흐름
1. Raw PDF 저장 (`data/raw`)
2. 전처리 및 chunk 생성 (`src/data/pdf_preprocessor.py`)
3. 임베딩 + Chroma 저장 (`src/ingest/build_index.py`)
4. 질의 시 하이브리드 검색 (`src/retrieval/hybrid_retriever.py`)
5. 검색 결과 기반 답변 + 출처 (`src/rag/qa_chain.py`)

## 3. 환각 방지 설계
- 시스템 프롬프트에 "컨텍스트 외 정보 금지" 명시
- 답변 말미에 출처 강제 포맷 포함
- 검색 실패 시 모름 응답 정책

## 4. 확장 포인트
- Metadata Filtering: 지역/연령/업력 조건 사전 필터링
- Reranker 도입: 교차 인코더 재순위화
- Multi-Vector Retriever: 제목/표/본문 분리 임베딩
