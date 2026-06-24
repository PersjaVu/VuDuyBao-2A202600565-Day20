# Screenshots

Đặt ảnh chụp màn hình bằng chứng vào đây để nộp kèm.

## Cần chụp

1. **`langsmith_trace.png`** — trace trên LangSmith (bắt buộc theo đề).
   - Vào https://smith.langchain.com → project **`VinUni`** → mở run `workflow` mới nhất.
   - Chụp trace tree: `workflow → agent:researcher / analyst / writer / critic → llm:*`
     kèm cột **latency** và **tokens** của từng bước.

2. (Tuỳ chọn) **`benchmark.png`** — bảng kết quả `python -m multi_agent_research_lab.cli benchmark`
   hoặc nội dung [../reports/benchmark_report.md](../reports/benchmark_report.md).

3. (Tuỳ chọn) **`cli_run.png`** — output lệnh `multi-agent` (câu trả lời + route + critic verdict).

> Nếu nộp bằng link thay vì ảnh, dán link share của run LangSmith vào đây hoặc vào SOLUTION.md.
