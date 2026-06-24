# Failure modes & fixes — Multi-Agent Research Lab

Giải thích các failure mode của hệ multi-agent và cách repo này phòng/chữa (defense in depth).
Mỗi mục: **triệu chứng → nguyên nhân → cách fix trong code → file liên quan**.

---

## 1. Runaway loop (vòng lặp vô tận)

- **Triệu chứng:** Supervisor cứ định tuyến qua lại giữa các agent, không bao giờ dừng → đốt
  token và thời gian vô hạn.
- **Nguyên nhân:** routing policy không có điều kiện dừng cứng; một agent không tạo ra output
  mà nó "phải" tạo nên bước sau không bao giờ đủ điều kiện.
- **Fix:**
  - `max_iterations` (mặc định 6) kiểm tra trong `SupervisorAgent.decide` → đạt ngưỡng là trả
    `done`, vòng lặp **luôn kết thúc**.
  - Routing theo thứ tự ưu tiên rõ ràng (research → analysis → answer → review → done), mỗi
    bước chỉ chạy khi field tương ứng còn thiếu.
- **File:** [supervisor.py](src/multi_agent_research_lab/agents/supervisor.py),
  [workflow.py](src/multi_agent_research_lab/graph/workflow.py)

## 2. Stuck / slow worker (agent treo hoặc gọi LLM quá lâu)

- **Triệu chứng:** một call LLM treo hoặc rất chậm (model reasoning `gpt-oss-120b` mất
  ~20–40s/call) → tổng thời gian vượt ngân sách. Thực tế đã gặp: với timeout mặc định 60s, một
  run hợp lệ 3 call bị guardrail timeout cắt ngang.
- **Nguyên nhân:** không có giới hạn wall-clock; latency thật của provider cao hơn dự kiến.
- **Fix:**
  - `timeout_seconds` (deadline wall-clock) trong `MultiAgentWorkflow.run`. Đã **tinh chỉnh
    logic**: chỉ abort khi *còn việc phải làm*, nên run kết thúc tự nhiên ngay sát mốc thời gian
    không bị gắn cờ timeout oan.
  - Nâng ngân sách cấu hình lên 300s trong `.env` để hợp với model chậm.
  - `timeout` được truyền vào client OpenAI ở mỗi request.
- **File:** [workflow.py](src/multi_agent_research_lab/graph/workflow.py),
  [config.py](src/multi_agent_research_lab/core/config.py)

## 3. Agent fail liên tục (exception lặp lại)

- **Triệu chứng:** một agent ném lỗi mỗi lần chạy; nếu cứ retry mãi sẽ kẹt.
- **Nguyên nhân:** lỗi dữ liệu/đầu vào khiến agent đó luôn fail.
- **Fix (retry + fallback):**
  - Exception được bắt trong workflow, đếm vào `state.failures[agent]`, ghi `agent_error` vào
    trace.
  - Sau `_MAX_AGENT_FAILURES` (2) lần, Supervisor **định tuyến vòng qua** agent hỏng thay vì
    retry vô hạn → hệ vẫn tiến tới `done`.
- **File:** [workflow.py](src/multi_agent_research_lab/graph/workflow.py),
  [supervisor.py](src/multi_agent_research_lab/agents/supervisor.py)

## 4. Provider lỗi tạm thời (LLM/search rớt mạng)

- **Triệu chứng:** một call API fail do mạng/rate-limit nhất thời.
- **Fix:**
  - **LLM:** `tenacity` retry (3 lần, exponential backoff) trong `LLMClient._complete_live`.
  - **Search:** thứ tự Tavily → DuckDuckGo (thật) → corpus local làm dự phòng cuối nếu cả hai
    backend thật không khả dụng, nên một blip mạng không làm sập cả run.
- **File:** [llm_client.py](src/multi_agent_research_lab/services/llm_client.py),
  [search_client.py](src/multi_agent_research_lab/services/search_client.py)

## 5. Silent fake output (trả dữ liệu giả mà không báo)

- **Triệu chứng:** thiếu API key nhưng hệ vẫn "trả lời" bằng text bịa → nguy hiểm vì khó phát
  hiện.
- **Fix:** `LLMClient.complete` **raise `LabError` rõ ràng** khi không có credential, thay vì
  fallback mock. Không có mock LLM trong code sản phẩm; test dùng `FakeLLM` inject qua DI.
- **File:** [llm_client.py](src/multi_agent_research_lab/services/llm_client.py)

## 6. Hallucination / claim không có nguồn

- **Triệu chứng:** câu trả lời nghe hợp lý nhưng không dẫn nguồn, hoặc bịa số liệu.
- **Fix:**
  - Writer luôn gắn **Sources map** ánh xạ mỗi marker `[n]` về một tài liệu thật đã thu thập.
  - Đo **citation coverage** = `cited_claims / total_claims` (đếm marker `[n]` thật trong câu
    trả lời).
  - **Critic agent** kiểm tra coverage + marker hallucination (`guarantee`, `100%`, `always`…)
    và đưa verdict PASS/REVIEW.
- **File:** [writer.py](src/multi_agent_research_lab/agents/writer.py),
  [critic.py](src/multi_agent_research_lab/agents/critic.py),
  [utils/text.py](src/multi_agent_research_lab/utils/text.py)

## 7. Workflow kết thúc mà không có câu trả lời

- **Triệu chứng:** mọi agent fail / timeout → không có `final_answer` → trả về rỗng.
- **Fix (validation post-condition):** `_validate` trong workflow tổng hợp một **câu trả lời
  fallback** mô tả tiến độ một phần (số nguồn, có research/analysis chưa); nếu thất bại mà
  không có lỗi nào được ghi thì raise `AgentExecutionError`. Hệ **không bao giờ trả về rỗng**.
- **File:** [workflow.py](src/multi_agent_research_lab/graph/workflow.py)

## 8. Mất context khi handoff giữa các agent

- **Triệu chứng:** agent sau không biết agent trước đã làm gì → lặp việc hoặc mất thông tin.
- **Fix:** một `ResearchState` duy nhất là **single source of truth** đi qua mọi bước, với field
  riêng cho từng output (sources, research_notes, analysis_notes, final_answer, critic_review)
  cộng `agent_results` + `trace` để tái dựng toàn bộ luồng.
- **File:** [state.py](src/multi_agent_research_lab/core/state.py)

---

## Tổng kết phòng thủ nhiều lớp

| Lớp | Cơ chế |
|---|---|
| Giới hạn vòng lặp | `max_iterations` |
| Giới hạn thời gian | `timeout_seconds` (chỉ cắt khi còn việc) |
| Chịu lỗi agent | retry-count + route-around fallback |
| Chịu lỗi provider | tenacity retry; search nhiều backend |
| Không giả mạo | thiếu key → raise, không mock |
| Chống hallucination | Sources map + citation coverage + Critic |
| Đảm bảo đầu ra | `_validate` fallback answer |
| Không mất context | shared `ResearchState` + trace |

Trace của mỗi run (kể cả lúc lỗi) xem được trên **LangSmith** (project `VinUni`) và bản JSON
local qua `--trace-out` — giúp giải thích "ai làm gì, tốn bao nhiêu, sai ở đâu".
