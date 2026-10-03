# Lộ trình học và xây

Mỗi nhiệm vụ nên có: phần học liên quan, đầu ra kiểm tra được, ghi chú điều đã học và một commit phù hợp. Các mục chưa đánh dấu đều chưa triển khai.

## v0.1 — Tìm kiếm tài liệu (phần 1–3)

- [ ] 01: Đọc notebook 1–4, trích markdown và code hữu ích. Giữ notebook, cell ID và chỉ số cell trong metadata. Kiểm tra không có ảnh nhúng, outputs hoặc cell thông tin xác thực.
- [ ] 02: Chia nội dung thành các đoạn. Kiểm tra đoạn mẫu và metadata; giải thích ảnh hưởng của chunk size và overlap.
- [ ] 03: Chọn embedding và vector store, tạo chỉ mục. Lưu cấu hình và phiên bản dependencies đã kiểm tra.
- [ ] 04: Nhập câu hỏi và xem top k đoạn cùng nguồn. Dùng ít nhất 5 câu hỏi để kiểm tra tìm kiếm.

## v0.2 — Hỏi đáp có dẫn nguồn (phần 4)

- [ ] Ghép context vào prompt và tạo câu trả lời.
- [ ] Hiển thị notebook, vị trí nội dung và đoạn bằng chứng; chỉ dẫn các nguồn đã truy xuất.
- [ ] Kiểm tra trường hợp tài liệu không đủ thông tin.
- [ ] Tạo 15 câu hỏi có đáp án và 5 câu ngoài phạm vi; ghi nguồn bằng chứng. Giữ một phần bộ câu hỏi để đánh giá cuối.

## v0.3 — Cải thiện truy xuất (phần 5–9)

- [ ] Bổ sung nội dung notebook 5–9 và cập nhật bộ đánh giá.
- [ ] Chạy cấu hình cơ bản trên bộ câu hỏi để có số liệu đối chiếu.
- [ ] Thử Multi Query và RAG-Fusion riêng từng phương pháp.
- [ ] Thử Decomposition, Step Back hoặc HyDE khi có loại lỗi phù hợp.
- [ ] Ghi lợi ích, trường hợp suy giảm, thời gian và số lần gọi model. Giữ phương pháp có bằng chứng hữu ích.

## v0.4 — Lọc và xếp hạng (phần 10–11, 15)

- [ ] Bổ sung metadata bài học và chủ đề.
- [ ] Thử metadata filters; routing khi có nhiều nguồn cần phân luồng.
- [ ] Thử reranking và so sánh với cấu hình cơ bản.

## v1.0 — Demo và tài liệu

- [ ] Thêm giao diện hỏi đáp đơn giản, cho xem bằng chứng.
- [ ] Viết hướng dẫn cài đặt và chạy bằng môi trường mới.
- [ ] Cung cấp dữ liệu mẫu có nguồn và quyền sử dụng phù hợp.
- [ ] Công bố kết quả đánh giá và giới hạn quan sát được.

## Thử nghiệm tiếp theo

Phần 12–14 và 16–18: Multi-representation Indexing, RAPTOR, ColBERT, CRAG, Self-RAG và long context. Chọn theo vấn đề thực tế sau khi có baseline; không phải điều kiện để hoàn thành phiên bản đầu.
