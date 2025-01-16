# Nhận diện từ tiếng Việt bằng mô hình PARSeq
## 1. Cài đặt các thư viện cần thiết
```shell
pip install -r requirements.txt
```

## 2. Chuyển file `.ckpt` sang `.pth`
```python
from parseq import PARSeq
PARSeq.ckpt2pth(ckpt_path="<path_to_ckpt>",
                pth_path="<path_to_pth>")
```
Các mô hình đã được tune có thể tìm thấy tại [đây](https://drive.google.com/drive/folders/1XUvxMZxACfx2xcJn80edargftvvUz9rY?usp=drive_link).

## 3. Inference ảnh
```python
from parseq import PARSeq
from PIL import Image

# `variant` can be either `parseq` or `parseq-tiny`
parseq = PARSeq.get_model(variant='parseq-tiny', state_dict_file="<path_to_pth>")

# Single image
img = Image.open("<path_to_img>")
output_str, confident = parseq(img) # Always return 2 lists regradless of the number of images

# Multiple images/batch inference
imgs = [Image.open("<path_to_img1>"), Image.open("<path_to_img2>"), ...]
output_strs, confidents = parseq(imgs)
```