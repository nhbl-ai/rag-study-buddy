# RAG Study Buddy

Ứng dụng chat trả lời câu hỏi dựa trên tài liệu được cung cấp, kèm nguồn và đoạn bằng chứng. Khi tài liệu không đủ thông tin, ứng dụng cần nói rõ điều đó.

## Trạng thái

Repo là khung ban đầu để tự code và build. Chưa có ứng dụng chạy được, dependencies hoặc lệnh khởi động. Chưa chọn model, vector store hay giao diện.

## Cấu trúc

```text
src/rag_study_buddy/       Khung mã nguồn Python
docs/project-context.md   Bối cảnh và quyết định dự án
AGENTS.md                 Hướng dẫn cộng tác
```

Bổ sung cấu trúc, dependencies và hướng dẫn chạy khi triển khai. Tài liệu đầu vào có thể được chọn theo nhu cầu; không bắt buộc dùng notebook hoặc theo lộ trình học nào.

## Dữ liệu cục bộ

`.env`, `data/raw/`, `data/processed/`, `data/private/` và các thư mục dữ liệu sinh ra đã có trong `.gitignore`. Chỉ thêm dữ liệu mẫu được phép chia sẻ vào Git.
