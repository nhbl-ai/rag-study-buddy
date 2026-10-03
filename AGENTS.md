# Hướng dẫn cộng tác

## Mục tiêu và ngôn ngữ

- Dự án giúp người dùng vừa học theo `langchain-ai/rag-from-scratch` vừa xây trợ lý hỏi đáp tài liệu có dẫn nguồn.
- Trao đổi và giải thích bằng tiếng Việt. Tên biến, hàm và module dùng tiếng Anh dễ hiểu.
- Đọc `docs/project-context.md` và `docs/roadmap.md` trước khi tiếp tục một chat mới.

## Cách học và triển khai

- Ưu tiên một bước nhỏ có thể kiểm tra, gắn với phần học tương ứng. Giải thích mục đích, đầu vào, đầu ra và cách kiểm tra.
- Theo phạm vi người dùng yêu cầu: nếu đang học một bước, không tự triển khai toàn bộ các mốc sau. Khi người dùng yêu cầu làm trọn một nhiệm vụ, hoàn thành nhiệm vụ đó.
- Bắt đầu với notebook 1–4. Giữ tên notebook, cell ID hoặc chỉ số cell để truy xuất về nguồn.
- Đọc notebook dưới dạng JSON; bỏ ảnh nhúng, outputs và cell cấu hình chứa thông tin xác thực khỏi tập dữ liệu tra cứu.
- Phân biệt trích xuất dữ liệu, chia đoạn, retrieval và generation để dễ quan sát lỗi.
- Đối chiếu tài liệu chính thức khi chọn phiên bản và API; repo nguồn có import cũ. Khóa phiên bản sau khi đã kiểm tra môi trường chạy được.

## Kiểm tra và ghi tiến độ

- Thực hiện kiểm tra phù hợp với thay đổi. Không báo đã chạy ứng dụng hoặc API nếu chưa thực hiện.
- Đánh giá retrieval riêng với generation. Không khẳng định cải thiện chỉ từ một ví dụ.
- Cập nhật trạng thái trong `docs/roadmap.md` khi hoàn thành nhiệm vụ; ghi các thử nghiệm vào `docs/experiments.md`.
- Không commit API key, `.env`, dữ liệu riêng tư, vector store hoặc outputs lớn. Không sửa mã bên thứ ba chỉ để che lỗi.
