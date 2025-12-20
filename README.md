# VNPT Hackathon 2025

## Pipeline Flow (predict.py)
![Workflow](img/workflow.jpg)

Our system uses an intelligent **Adaptive Pipeline** architecture, automatically routing questions to the most optimal processing workflow instead of applying a rigid RAG approach for every case.

### 1. Question Classification (Adaptive Routing)
Every input Query (Question + Choices) is classified by **LLM Small** into 4 strategic groups:

*   **🛡️ Safety & Policy Filter (`cannot_answer`)**:
    *   Identifies sensitive, policy-violating, or harmful questions.
    *   **Action**: The system immediately selects a refusal answer (e.g., "I cannot answer...") without further processing, ensuring absolute safety.

*   **🧮 Advanced STEM Reasoning (`calculation`)**:
    *   Dedicated to Math, Physics, and Chemistry questions requiring precise calculations.
    *   **"Expert-Auditor" Model**:
        1.  **Stage 1 (The Expert - LLM Large)**: Analyzes the problem, sets up formulas, solves step-by-step, and provides an initial answer.
        2.  **Stage 2 (The Auditor - LLM Large)**: Acts as an auditor, double-checking the expert's logic and calculations to ensure no arithmetic "hallucinations" before finalizing the answer.

*   **📖 Context-Aware Reading (`has_context`)**:
    *   Dedicated to reading comprehension questions that already include a long text passage in the prompt.
    *   Uses **LLM Large** with a specialized prompt for reading comprehension skills, focusing on exploiting internal data without triggering RAG to avoid external information noise.

*   **🌐 Adaptive RAG (`general`)**:
    *   Dedicated to general knowledge questions requiring external information lookup.
    *   **Advanced RAG Process**:
        1.  **Query Expansion**: Combines `Question + Choices` to increase search context.
        2.  **Hybrid Search**: Combines **Dense Embedding** (VNPT API) and **Sparse Embedding** (BM25) on Qdrant, using the **Reciprocal Rank Fusion (RRF with k = 60)** algorithm to retrieve the Top 30 potential documents.
        3.  **LLM-as-a-Judge Reranking**: Instead of using a standard Cross-Encoder, the system uses **LLM Small** to "score" 30 documents based on 4 criteria: *Topic Relevance, Keyword Presence, Information Usefulness, and Freshness*.
        4.  **Filtering**: Retains only a maximum of **Top 5** documents with a score > 7.0.
        5.  **Final Answer**: **LLM Large** synthesizes information from these high-quality documents to provide the final answer.

> **Note**: Using `LLM Small` as a Reranker allows processing a larger Context Window (evaluating 30 documents at once) and is more flexible in assessing information "freshness" compared to traditional Rerank models. All LLMs use `temperature = 0` to avoid Hallucinations.

## Data Processing
- Data sources and processing methods are available at: [sheet](https://docs.google.com/spreadsheets/d/176Hs2OUBhQj6UrNkRyu4dse9xK_ag_6VMZZsd2maLy8/edit?usp=sharing)
- Aggregated data is available at: [Data](https://drive.google.com/drive/folders/1dUnodqUE3Ea0ESjACpEEGARFgtiua-KR?usp=sharing). In this folder, `input` contains crawled files, and `output` contains files converted to `.csv` format.
- Data is crawled from various sources. After crawling, data is merged into a single `.csv` file with 3 columns: "id", "title", and "text". The "id" and "title" columns are not critical, while the "text" column contributes directly to the vector database.
- Finally, everything is merged into a single file: `data.csv`. You can donwload it from `data.zip` file in the drive link and put it in folder `vectorDB/data`.

## Vector Database work flow (High-Performance Indexing)

We have built a highly optimized **Asynchronous Indexing Pipeline** (`vectorDB/main_async.py`) to process large amounts of data quickly and robustly:

1.  **Hybrid Search Architecture**:
    *   Combines the power of **Dense Embedding** (VNPT AI API, 1024 dim) to capture semantics and **Sparse Embedding** (FastEmbed BM25) to capture exact keywords.
    *   Configures the Qdrant collection with both `vectors_config` (Cosine) and `sparse_vectors_config` (IDF), laying the foundation for a high-precision Hybrid Search algorithm.

2.  **Parallel Async Processing (Creative & Optimized)**:
    *   Instead of running sequentially, the system uses `asyncio` to execute the two heaviest tasks in parallel: **Calling Embedding API** and **Calculating BM25** simultaneously (`asyncio.gather`).
    *   Increases processing speed by **5-8 times** compared to the traditional synchronous version.

3.  **Robust API Rate Limiting & Resilience**:
    *   Designed a **Semaphore** mechanism to strictly control the number of concurrent requests (`MAX_CONCURRENT_REQUESTS`), ensuring the 500 req/min limit of VNPT API is never exceeded.
    *   Integrated intelligent **Exponential Backoff** mechanism: automatically waits and retries when encountering network errors or 429 Too Many Requests.
    *   **Note**: When restarting from scratch, you can comment out `.sleep` lines to run faster.

4.  **Smart Resume & Deduplication**:
    *   The system automatically scans for existing Point IDs in Qdrant before running.
    *   Allows pausing and resuming the indexing process at any time without restarting from the beginning, saving API costs and time.

5.  **Context-Aware Chunking**:
    *   Uses `SentenceSplitter` from Llama-index to split text by sentence semantics (`chunk_size=512`, `overlap=32`), avoiding cutting in the middle and losing context.

**Storage Data Structure (Payload):**
```json
{
  "id": "point_id (generated sequentially)",
  "vector": {
    "dense": [0.01, -0.02, ...], // VNPT Embedding
    "sparse": {
      "indices": [12, 50, ...],   // BM25 Indices
      "values": [0.5, 0.8, ...]   // BM25 Values
    }
  },
  "payload": {
    "doc_id": "Original Document ID",
    "title": "Document Title",
    "chunk_index": "Index of chunk in document",
    "text": "Full text content of the chunk"
  }
}
```

## Resource Initialization
### API Keys Configuration
The `api-keys.json` file contains API credentials for VNPT AI Services:

```json
[
  {
    "authorization": "Bearer <JWT_TOKEN>",
    "tokenKey": "<PUBLIC_KEY_BASE64>",
    "llmApiName": "<MODEL_NAME>",
    "tokenId": "<TOKEN_UUID>"
  }
]
```

| Field | Description |
|-------|-------------|
| `authorization` | JWT Bearer token for API authentication |
| `tokenKey` | RSA public key (Base64) for encryption |
| `llmApiName` | Model name: `LLM large`, `LLM small`, or `LLM embedings` |
| `tokenId` | Token UUID identifier |

> **Note:** Configure `api-keys.json` with valid credentials before running `predict.py` and `vectorDB/main_async.py`. This file is in the main folder.

### How to run
1. Download file `data.csv` in `data.zip` at this [drive](https://drive.google.com/drive/folders/1dUnodqUE3Ea0ESjACpEEGARFgtiua-KR?usp=sharing). Put it in folder `/vectorDB/data`
2. Install Dependencies!
```
pip install -r requirements.txt
```
3. Run the Qdrant vector database:
```
docker-compose up -d
```
4. Run the Qdrant vector database workflow:
```
cd vectorDB
python main_async.py
```
5. Ensure file `private_test.json` is in the main folder. Run the workflow pipeline:
```
python main.py
```
6. `submission.csv` and `predict.json` and `submission_time.csv is in folder `output`
7. Note:
- `main_async.py`: To follow the api rate limit (500 req/m), **UNCOMMENT**  lines that have term `.sleep`. Ensure there is no error in file `.log`
- `predict.py`: To follow the api rate limit (40 req/h for LLM Large and 60 req/h for LLM Small), **UNCOMMENT**  lines that have term `.sleep`. Ensure there is no error in file `.log`. If there is a line say **Default to A** in file `.log`, please run that test case again.

### How to run with docker

We have pre-built the Docker image and pushed it to Docker Hub. You can easily pull and run the solution without building it locally.

1.  **Pull the image:**

    ```bash
    docker pull khoadeptraivai/vnpt-hackathon-submission:v1
    ```

2.  **Run the container:**

    Ensure you are in the project root directory where `private_test.json` is located.

    ```bash
    docker run --rm -v ${PWD}/output:/code/output -v ${PWD}/private_test.json:/code/private_test.json khoadeptraivai/vnpt-hackathon-submission:v1
    ```

    **Explanation:**
    *   `--rm`: Automatically remove the container when it exits.
    *   `-v ${PWD}/output:/code/output`: Mounts your local `output` folder to `/code/output` in the container. This ensures that `submission.csv` and other results are saved to your machine.
    *   `-v ${PWD}/private_test.json:/code/private_test.json`: Mounts your local input file `private_test.json` to the container so the code can read it.

    > **Note**: If you want to build the image locally instead, you can run:
    > ```bash
    > docker build -t khoadeptraivai/vnpt-hackathon-submission:v1 .
    > ```

