from PIL import Image
from parseq import PARSeq

img_path = r"<path to image>"
pth_path = "<path to pth"
variant = "parseq-tiny" # Either "parseq-tiny" or "parseq"
device = "cpu"

img = Image.open(img_path).convert("RGB")
model = PARSeq.get_model(variant=variant, state_dict_file=pth_path, device=device)

result = model(img)
print(result)