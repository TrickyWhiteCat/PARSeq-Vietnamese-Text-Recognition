# Nhận diện từ tiếng Việt bằng [PARSeq](https://github.com/baudm/parseq)
Code dùng để nhận diện tiếng Việt trong văn bản (bao gồm cả các ký tự đặc biệt) sử dụng model [PARSeq](https://github.com/baudm/parseq) được finetune bằng 5 triệu ảnh synthetic tiếng Việt. Model expect input đầu vào là ảnh của 1-3 từ tiếng Việt đã được cắt ra bằng model text detection.

MODEL KHÔNG CÓ KHẢ NĂNG NHẬN DIỆN CẢ MỘT DÒNG KÝ TỰ DÀI.
## 1. Cài đặt các thư viện cần thiết
**Lưu ý:** Để tránh xung đột và đảm bảo khả năng tương thích với các phiên bản CUDA khác nhau, cần tự cài đặt pytorch version 2.5.1.
```shell
pip install -r requirements.txt
```

## 2. Khởi tạo model
Có thể khởi tạo model từ file checkpoint (`.ckpt`)...
```python
from parseq import PARSeq
parseq = PARSeq.load_from_checkpoint("<path_to_ckpt>")
```
hoặc load từ file state dict (`.pth`)
```python
# `variant` can be either `parseq` or `parseq-tiny`
parseq = PARSeq.get_model(variant='parseq-tiny', state_dict_file="<path_to_pth>")
```
Ngoài ra, có thể convert file `.ckpt` sau khi finetune sang file `.pth` để giảm dung lượng file (do file `.ckpt` ngoài lưu state dict còn lưu các thông tin của quá trình finetuning).
```python
from parseq import PARSeq
PARSeq.ckpt2pth(ckpt_path="<path_to_ckpt>",
                pth_path="<path_to_pth>")
```
Các mô hình đã được tune:
| Model Type | Model Size (MB) | Accuracy (%) | Download |
|---|---|---|---|
| Standard | 91.38 | 98.81 | [Link](https://github.com/TrickyWhiteCat/parseq-vietnamese/releases/download/models/standard_1.0.0.pth) |
| Tiny | 23.22 | 98.18 | [Link](https://github.com/TrickyWhiteCat/parseq-vietnamese/releases/download/models/tiny_1.0.0.pth) |

Các mô hình được đánh giá trên một bộ dữ liệu văn bản scan, bao gồm cả ký tự đặc biệt. Nếu bạn không cần ký tự đặc biệt, kết quả đánh giá khả năng cao sẽ tốt hơn. Trong trường hợp đó, bạn phải tự đánh giá xem model có đủ tốt cho usecase của bạn không.

Ảnh input đầu vào nên có chiều cao tối thiểu 20-30px để có kết quả tốt nhất. Ảnh có kích thước sẽ làm giảm độ chính xác của model.

## 3. Inference ảnh
```python
# Single image
img = Image.open("<path_to_img>")
output_str, confident = parseq(img)

# Multiple images/batch inference
imgs = [Image.open("<path_to_img1>"), Image.open("<path_to_img2>"), ...]
outputs = parseq(imgs)
```

Nếu có một list ảnh mà không muốn batch thủ công thì có thể inference bằng `model.auto_batch` tương tự như ví dụ sau:

```python
outputs = model.auto_batch(iterable=img_paths,
                           preprocess_fn=lambda img_path: Image.open(img_path).convert("RGB"), # Read img from path
                           postprocess_fn=lambda x: x[0], # Only get the output string
                           batch_size=128)
```
