# 창업지원 문서 기반 RAG 질의응답 시스템

이 프로젝트는 정부/지자체 창업지원 문서를 기반으로, 환각을 줄이고 출처를 명시하는 RAG 시스템을 구현합니다.

## 핵심 목표
- 문서를 벡터화하여 검색 정확도 향상
- 하이브리드 검색(BM25 + 벡터 검색) 적용
- 메타데이터 필터링(지역/창업단계/나이) 반영
- 답변에 출처(파일명/페이지) 강제 표기

## 폴더 구조
- `src/data`: PDF 로딩/전처리
- `src/ingest`: 임베딩 및 벡터 DB 적재
- `src/retrieval`: 하이브리드 검색
- `src/rag`: 답변 체인
- `src/eval`: 테스트 실행
- `docs`: 필수 산출물 문서

## 빠른 시작
1. 패키지 설치
```bash
pip install -r requirements.txt
```
2. 환경 변수 설정
```bash
cp .env.example .env
```
3. 웹 데이터 수집(선택)
```bash
python app.py collect-web
```
4. 원본 PDF를 `data/raw`에 저장
5. 인덱스 생성 (PDF + JSON 동시 반영)
```bash
python app.py build-index
```
6. 질의 실행
```bash
python app.py ask "만 39세 서울 거주 예비 창업자가 받을 수 있는 지원금은?"
```

비교 질의 예시:
```bash
python app.py ask "예비창업패키지와 청년창업사관학교 지원 항목 차이점을 알려줘"
```

실행 시 `적용 필터`가 함께 출력되며, 검색 단계에서 지역/단계/나이 조건을 우선 반영합니다.
7. 테스트 시나리오 실행
```bash
python app.py eval
```

원클릭 실행(수집 + 인덱스 + 평가):
```bash
python app.py run-all
```

## 웹 수집 데이터 포맷
`collect-web` 실행 시 `data/raw/web_seed.json` 파일이 생성됩니다.

레코드 스키마:
- `source_site`: 출처 사이트 키 (`k-startup`, `kised`, `kosmes`)
- `url`: 원문 URL
- `title`: 페이지 제목
- `body`: 정제 본문 텍스트
- `links`: 페이지 내 링크/공고 식별자 목록 (`title`, `url`, `notice_id`)

인덱싱 시 `links`는 별도 문서로 분해되어 공고 제목/URL/ID 기반 검색 정확도를 높입니다.

## 권장 데이터
- K-Startup 공고문 PDF
- 중소벤처기업부/소진공 정책자금 가이드
- 지자체 창업지원 공고/조례
