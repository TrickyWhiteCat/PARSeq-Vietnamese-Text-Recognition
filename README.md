# Nhận diện từ tiếng Việt bằng [PARSeq](https://github.com/baudm/parseq)
## 1. Cài đặt các thư viện cần thiết
**Lưu ý:** Để tránh xung đột và đảm bảo khả năng tương thích với các phiên bản CUDA khác nhau, cần tự cài đặt pytorch version 2.5.1.
```shell
pip install -r requirements.txt
```

## 2. Khởi tạo model
Có thể khởi tạo model từ file checkpoint (`.ckpt`)...
```python
from parseq import PARSeq
PARSeq.load_from_checkpoint("<path_to_ckpt>")
```
hoặc load từ file state dict (`.pth`)
```python
# `variant` can be either `parseq` or `parseq-tiny`
parseq = PARSeq.get_model(variant='parseq-tiny', state_dict_file="<path_to_pth>")
```
Ngoài ra, có thể convert file `.ckpt` sau khi finetune sang file `.pth` để giảm dung lượng file (do file `/ckpt` ngoài lưu state dict còn lưu các thông tin của quá trình finetuning).
```python
from parseq import PARSeq
PARSeq.ckpt2pth(ckpt_path="<path_to_ckpt>",
                pth_path="<path_to_pth>")
```
Các mô hình đã được tune có thể tìm thấy tại [đây](https://drive.google.com/drive/folders/1XUvxMZxACfx2xcJn80edargftvvUz9rY?usp=drive_link).

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
