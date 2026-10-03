# RAG Study Buddy

Vừa học RAG vừa xây một trợ lý tra cứu tài liệu kỹ thuật có dẫn nguồn.

Nguồn học: [langchain-ai/rag-from-scratch](https://github.com/langchain-ai/rag-from-scratch).
Đây là dự án thực hành riêng; các tài liệu học tham chiếu về nguồn gốc.

## Mục tiêu

Người dùng hỏi về RAG, ứng dụng tìm các đoạn liên quan rồi trả lời kèm notebook và bằng chứng. Khi tài liệu không đủ thông tin, ứng dụng cần nói rõ điều đó.

Ví dụ:

- Indexing, retrieval và generation khác nhau thế nào?
- Đoạn code nào thực hiện retrieval trong bài 1–4?
- Multi Query và RAG-Fusion khác nhau thế nào? (Sau khi thêm bài 5–9.)

## Trạng thái hiện tại

Repo đã có cấu trúc, lộ trình và hướng dẫn cộng tác. Chưa có ứng dụng chạy được; chưa cài thư viện hoặc gọi API trả phí.

Mốc đầu tiên là **v0.1 — tìm kiếm tài liệu từ notebook 1–4**. Bắt đầu bằng việc đọc notebook và giữ thông tin nguồn, sau đó mới chia đoạn và tìm kiếm.

## Bắt đầu học

1. Đọc [bối cảnh dự án](docs/project-context.md).
2. Chọn nhiệm vụ đầu tiên trong [lộ trình](docs/roadmap.md).
3. Học phần tương ứng trong repo nguồn, tự giải thích luồng dữ liệu rồi triển khai một bước nhỏ.
4. Kiểm tra kết quả, ghi điều đã học và commit.

## Cấu trúc

```text
src/rag_study_buddy/   Mã ứng dụng khi bắt đầu triển khai
notebooks/            Thí nghiệm và ghi chú khi học
data/sample/          Dữ liệu mẫu và thông tin nguồn
evals/                Bộ câu hỏi đánh giá
docs/                 Bối cảnh, lộ trình và kết quả thí nghiệm
AGENTS.md             Hướng dẫn cho Codex khi làm việc trong repo
```

## Đánh giá

Chuẩn bị 15 câu hỏi có đáp án trong tài liệu và 5 câu ngoài phạm vi. Với câu có đáp án, ghi đáp án kỳ vọng và vị trí bằng chứng. Đánh giá riêng chất lượng truy xuất và chất lượng câu trả lời.

Mỗi lần thử một kỹ thuật mới, giữ nguyên dữ liệu và cấu hình còn lại; ghi kết quả trong [nhật ký thí nghiệm](docs/experiments.md).

## Làm việc trong Codex

Thêm thư mục clone làm project local và mở chat trong project đó. Chat mới nên đọc `AGENTS.md`, `docs/project-context.md` và `docs/roadmap.md` trước khi bắt đầu.

Để tiếp tục buổi đầu, gửi:

> Đọc AGENTS.md và docs/. Chúng ta bắt đầu nhiệm vụ 01 của v0.1. Hãy hướng dẫn tôi đọc notebook 1–4 và xây bước trích xuất dữ liệu, giải thích từng bước để tôi thực hành.

## Thông tin riêng tư

API key đặt trong `.env` cục bộ; file này được bỏ qua bởi Git. Chỉ commit dữ liệu mẫu được phép chia sẻ. Chưa chọn license cho dự án; việc tham chiếu repo nguồn không cấp quyền phân phối lại tài liệu của bên thứ ba.
