# Tái lập hai bài preprocessing cho các task gốc

Mỗi task dùng một NPP riêng. Đây không phải mô hình joint AR+OD của các thí nghiệm trước.

| Bài | Task chạy | Analyzer đại diện | Codec đánh giá |
|---|---|---|---|
| Preprocessing Enhanced Image Compression for Machine Vision | COCO object detection | Faster R-CNN ResNet50 FPN | BPG |
| Cùng bài ảnh | ImageNet image classification | ResNet50 | BPG |
| A Preprocessing Framework for Video Machine Vision under Compression | Kinetics400 action recognition | Swin3D-T | x264/x265 medium |
| Cùng bài video | GOT10k object tracking | DiMP50 | x264/x265 medium |

Người dùng đã chốt QP **30/35/40/45** cho tất cả các run. Đây là thay đổi có chủ đích so với lưới QP gốc. Không có teacher bank, distillation, quality rollback hoặc identity fallback. Codec-only chỉ là mốc đo, không phải mô hình được xuất khi kết quả kém.

## Cơ chế

- Bài ảnh: NPP pixel1×1 + U-Net từ mã tác giả; proxy Minnen2018 autoregressive/hyperprior; forward dùng reconstruction/rate BPG thật, backward qua proxy; loss `R + λ_task L_task + .5 MSE(X,Y)`.
- Bài video: spatial/temporal convolution + conditional attention + RGB residual; virtual codec prediction/residual/DCT/quantization/factorized-CDF entropy; loss `10 [MSE(X,Xvirtual) + .001 Rvirtual] + Ltask`; train bằng virtual codec, đánh giá bằng codec thật.
- Freeze analyzer weights. Image proxy weights freeze trong các pha NPP, giữ quantization-noise surrogate khả vi. Video prior được fit trên TRAIN rồi freeze.

## Mức độ tái lập và phần chưa tương đương nguyên bản

Image NPP source pin: [XingtongGe/PreprocessingICM](https://github.com/XingtongGe/PreprocessingICM/tree/6a77c4633cd5bb6750c77ea9d213d38ffc88ee7a), Apache2.0; nguồn giữ trong `reproduce/vendor/` kèm LICENSE. Source QA điều kiện hóa bằng task-lambda/8; vì vậy không trình bày nó là raw-QP conditioning đúng từng chi tiết Figure3. Các lambda ở QP mới được nội suy log từ bảng paper. TableI proxy QP=[28,31,34,37,41,44], λp=[4096,2048,1024,512,256,128].

Mã tác giả và mô tả bài ảnh có khác biệt. Proxy dùng bản Minnen2018 pretrained từ CompressAI thay vì checkpoint proxy riêng chưa được phát hành; proxy được fit lại với BPG thật. ResNet50/FasterRCNN/Swin dùng Torchvision, không phải đúng implementation/checkpoint OpenMMLab của bài.

Bài video chưa có code tác giả được xác minh trong lượt triển khai này. Figure2/3 thiếu thông số và có annotation channel mâu thuẫn. Width16, temporal kernel3, một residual block/branch, sigmoid channel attention, DCT8, intra DC.5,9 motion candidates, STE và khởi tạo/fit factorized prior là **lựa chọn triển khai công khai**, không nhận là mã tác giả chính xác. DiMP dùng code/checkpoint chính thức; NPP tracking task loss dùng classification-score MSE + IoU-head MSE vì bài không định nghĩa LAcc chi tiết.

Tracking TRAIN chọn crop có reference nhìn thấy đối tượng và ít nhất một test frame nhìn thấy; loss task chỉ dùng các test frame có `cover>0`. Native PreciseRoIPooling giữ kernel gốc, materialize symlink từ ZIP trước compile, chuyển API THC sang C10; import backbone cũ được đổi từ `model_urls` sang enum weights V1. Mọi sửa tương thích được ghi trong `assets.json`.

Kinetics TRAIN hiện có là subset5percent, không phải toàn bộ K400 train. Validation dùng `jizeyong/kinetics400-rgb256-f10-val-data`: JPEG được ghép bằng YouTube ID với CSV validation chính thức của CVDF, pin SHA256 `358eaf47e7f80ebf9b17d49eb0635ad5e0fdab98a9cbd75ffdd2ee5d5e5b6944`. Cadence của bản JPEG chưa được xác minh; không tự nhận tương đương video nguyên bản. GOT10k đánh giá local trên validation có GT; không giả lập hidden TEST. Ảnh/video được lấy mẫu/resize theo config; các lựa chọn được giữ giống nhau khi so sánh codec-only/NPP.

## Chạy

```bash
pip install -e '.[cloud,dev]'
python -m reproduce.run --task od --mode pilot --inputs /kaggle/input --out /kaggle/working/paper_results
python -m reproduce.run --task ic --mode full --inputs /kaggle/input --out /kaggle/working/paper_results
python -m reproduce.run --task ar --mode full --inputs /kaggle/input --out /kaggle/working/paper_results
python -m reproduce.run --task ot --mode full --inputs /kaggle/input --out /kaggle/working/paper_results
```

Image full schedule:400k proxy updates toàn bank4QP,120k fixed-QP35,100k adaptive. Bộ đếm global proxy được khai báo rõ vì paper không nói rõ400k tính cho từng proxy hay toàn bộ. Video100k NPP steps là ngân sách triển khai, không phải số bước được bài video công bố. `pilot` dùng lịch ngắn để kiểm tra pipeline và **không được gọi là kết quả tái lập đầy đủ**.

Mỗi job kiểm tra thực trên GPU: đúng source hash, dữ liệu/nhãn, checkpoint pretrained, stream thật ở4QP, gradient task→proxy→NPP hữu hạn và trạng thái analyzer không thay đổi. Pha huấn luyện chỉ bắt đầu sau preflight này.

Kaggle notebooks luôn private và GPU-enabled; không nhúng credential. Khi hết ngân sách phiên, `last.pt` giữ NPP/proxy, optimizer, phase counters và RNG để tiếp tục. Nếu chưa có cập nhật NPP, không xuất `trained_npp.pt` và không đưa model khởi tạo vào metric thay thế.

AR R3 đã hoàn thành100k NPP steps nhưng validation bị dừng khi gặp JPEG sequence dưới31frame. `reproduce.ar_validation` chạy riêng đánh giá từ checkpoint đã hoàn thành lịch, ràng buộc SHA256 file/source/config, data/analyzer identity và strict model weights; không tạo optimizer hoặc gọi training. Khai báo `ar_validation_short_clip=loop`: clip ngắn lặp chỉ số modulo số frame, vẫn lấy16frame với stride2; clip đủ dài giữ nguyên chỉ số center-crop trước đó. Cả codec-only và NPP dùng cùng nguồn/chỉ số; source rỗng vẫn báo lỗi. Chính sách này dựa trên `SampleFrames(out_of_bound_opt='loop')` của [MMAction2](https://github.com/open-mmlab/mmaction2/blob/main/mmaction/datasets/transforms/loading.py). Config và từng cache row lưu chính sách/chỉ số; cache cũ không trộn với lần đánh giá đã sửa. Đây là sửa protocol validation được công khai, không phải huấn luyện lại hoặc thay đổi checkpoint.

```bash
python -m reproduce.ar_validation --checkpoint /path/to/last.pt --checkpoint-sha256 CHECKPOINT_SHA256 --training-source-sha256 TRAINING_SOURCE_SHA256 --inputs /kaggle/input --out /kaggle/working/paper_results
```

Các trainer snapshot cũ ghi `last.pt` trước khi cập nhật cờ `completed`. Bộ đánh giá kiểm tra đủ prior512/adaptive100k và `selected_step=100k`; mô hình xuất cuối R3 được đối chiếu từng tensor với checkpoint. Lần đánh giá sau có thể tiếp tục cache bằng `--cache-from /path/to/previous/paper_results`, giữ đúng checkpoint và source đánh giá.

Các input Kaggle được pin theo version: COCO2, ImageNet train1/val1, K400 train4/JPEG val1, GOT10k1. Checkpoint config còn ràng buộc hash analyzer, danh sách source/nhãn/COCO box và CSV nhãn K400. Resume từ phiên bản code hoặc dữ liệu khác sẽ bị từ chối. Với input local tự cung cấp, cần bảo đảm nội dung đúng version đã khai báo; tool không tự hash toàn bộ hàng triệu file ảnh/video.

## Kết quả và giới hạn báo cáo

`run.json` tách `execution_complete` khỏi `scientific_complete`. Một notebook chạy xong hoặc lưu checkpoint không có nghĩa đủ lịch paper/toàn bộ dữ liệu/toàn bộ validation. `validation/metrics.json` chứa điểm thật từng QP, PCHIP BD-rate nếu hợp lệ; đường không đơn điệu/không có miền chất lượng giao được ghi invalid. Full mode dùng toàn bộ validation có sẵn; cache theo source/cohort/model/config cho phép resume giữa các phiên. COCO AP được tính trên prediction gộp cả cohort. Full tracking xử lý toàn sequence, dùng `cover.label>0`, loại frame khởi tạo, và gộp IoU theo frame như toolkit GOT10k; chỉ frame đầu cung cấp box GT cho tracker. Pilot dùng clip16frame trên8sequence và không tự nhận là server TEST.

Không cam kết lặp lại mức giảm15–20% trong paper. Mọi sai khác về source, data, backbone, bước, kích thước và validation đều được lưu trong config/receipt.

## Nguồn

- [Bài ảnh](https://doi.org/10.1109/TCSVT.2024.3441049), [source tác giả](https://github.com/XingtongGe/PreprocessingICM).
- [Bài video](https://arxiv.org/html/2512.15331v1).
- [DiMP code](https://github.com/visionml/pytracking/tree/7eb9e74bd3d40e29dbcec444902237da13de247b), precise ROI submodule070a7950db6a945e30e8e3296204a1e975f131e8. Native extension chỉ sửa API THC cũ sang C10, giữ kernel gốc.

Kiểm thử: `python -m pytest -q`. Trên Windows có thể cần chỉ định `--basetemp` bên trong workspace.
