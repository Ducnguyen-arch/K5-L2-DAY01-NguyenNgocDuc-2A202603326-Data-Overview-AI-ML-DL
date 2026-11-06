# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 2026-11-06

**Runtime Colab:** GPU

**Python / PyTorch / Ultralytics:** 3.x / 2.x / 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh** (image classification) không phải **một vật thể** riêng lẻ vì model đang phân loại nội dung tổng thể của ảnh thành 1 lớp. JSON không có bounding box để xác định vị trí.
3. Class list **đi kèm taxonomy/checkpoint** (`yolo11n-cls.pt`), và không phải tự model nghĩ ra.
4. Phải cần cả 3 vì `class_id` là để định danh cho lớp để xác định lớp chính xác, `class_name` cần có để người hiểu đấy là cái gì, `taxonomy_name` cho biết đang dùng bộ lớp phân loại nào. 
5. Guideline cần quy định rõ tiêu chí chọn một lớp đại diện cho toàn bộ bức ảnh. 
6.`score` chỉ thể hiện mức độ tin cậy của model đối với cái dự đoán này nên không đảm bảo dự đoán đó chính xác, nó không được người xác nhận theo guideline.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=person ; `score`=0.912625 ; `bbox_xyxy`=[385.83, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68
2. Lớp `person`, góc trên-trái khoảng (385.83, 69.24) pixel, góc dưới-phải khoảng (498.92, 348.92) pixel, chiều rộng khoảng 113.58 pixel, chiều cao khoảng 279.68 pixel.
3. `score` của prediction là 0.912625, `score` threshold ban đầu là 0.35. Với nguõng 0.20 - 0.60 thì với `score` = 0.912625 lớn hơn cả 2 ngưỡng nên predcition được giữ lại cả hai.
4. Người **reviewer** phải kiểm tra bounding box, kiểm tra class có đúng vật thể không, kiểm tra vật thể bị bỏ sót.
5. Người gán nhãn phải vẽ box bao quanh toàn bộ phần nhìn thấy của vật thể, sát với đường viền vật thể nhất có thể. Nếu vật thể bị che khuất, chỉ vẽ phần nhìn thấy và không tự đoán vẽ phần thiếu của vẩt thể.
6. Vật thể bị khuất ở đây là vật ở góc khuất bên mép bàn gần bound ảnh. mình chọn **guideline** đưa tình huống cụ thể lên Lab Coach (escalation) trước, nếu guideline chưa quy định cách xử lý vật thể bị che khuất. 

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-010 ; `class_name`=bowl ; `score`=0.372406 ; số điểm polygon=`polygon_point_count`=12 ; 3–5 điểm đầu của `polygon_xy`=[[155.0, 170.0], [155.0, 177.0], [156.0, 178.0], [156.0, 179.0], [161.0, 184.0]]
2. Polygon xác định đường biên của vật thể thông qua nhiều điểm toạ độ.
3. `instance_id` để định danh 1 đối tượng cụ thể trong ảnh. ví dụ `class_id`= 45 là để cho biết đối tượng thuộc Bowl và cũng không phải là ID để tracking.
4. Người gán nhãn phải vẽ polygon sát theo đường viền thực tế của vật thể, chỉ bao gồm phần vật thể nhìn thấy được.
5. Trên ảnh `kitchen`, 2 đồ vật sát nhau trên mặt bàn. Chọn escalation lên Lab Coach nếu guideline chưa có quy định xử lý.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh |1 nhãn class đại diện cho toàn bộ ảnh  | Ảnh traffic dự đoán là 'cab' score=0.510915  |Quan sát ảnh, chọn class theo guideline  | check nhãn có đúng trong ảnh và taxonomy  |
| Phát hiện vật thể | 1 hoặc nhiều bounding box  |Khó quyết định vật thể bị che khuất  | Vẽ box sát viền vật thể nhìn thấy rõ | Check box xem có bao trọn toàn bộ vật thể không |
| Instance segmentation | polygon từng vật thể | khó xác định biên chính xác của vật thể nhỏ,bé  |Vẽ đường biên chính xác viền vật thể  | Check bound vật thể  |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
- Chỉ dùng ảnh COCO công khai có sẵn trong notebook; không tải ảnh cá nhân, khuôn mặt, biển số hay dữ liệu nội bộ lên Colab/GitHub.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
- Dừng thao tác và báo Lab Coach / GV phụ trách lab.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
