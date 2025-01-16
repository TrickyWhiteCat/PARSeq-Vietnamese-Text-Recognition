import os
from PIL import Image, ImageDraw, ImageFont
from randomize import random_crop

def expand_bbox(bbox: tuple[int, int], expand_pixel: tuple[int, int, int, int]):
    expand_left, expand_top, expand_right, expand_bottom = expand_pixel
    x_min, y_min, x_max, y_max = bbox
    x_min = x_min - expand_left
    x_max = x_max + expand_right
    y_min = y_min - expand_top
    y_max = y_max + expand_bottom
    return [x_min, y_min, x_max, y_max]

def draw_box_word(word: str, draw_object: ImageDraw.ImageDraw, expand_pixel: tuple[int, int, int, int] = (0, 0, 0, 0), text_color: tuple[int, int, int] = (0, 0, 0), **kwargs):
    bbox = draw_object.font.getbbox(word)
    expanded = expand_bbox(bbox, expand_pixel)
    anchor = -expanded[0], -expanded[1]
    bbox = draw_object.textbbox(anchor, word)
    draw_object.text(anchor, word, fill=text_color)
    return word, bbox

def width_height(word: str, font: ImageFont.ImageFont, expand_pixel = [0, 0, 0, 0]):
     x_min, y_min, x_max, y_max = expand_bbox(font.getbbox(word), expand_pixel)
     return x_max - x_min, y_max - y_min
        
def draw_bboxes(bboxes: list, draw_object: ImageDraw.ImageDraw, color = (255, 0, 0)):
    for bbox in bboxes:
        draw_object.rectangle(bbox, outline=color)

def create_new(bg: str|tuple, content: str = None, draw_bbox: bool = False, expand_pixel: tuple[int, int, int, int] = (0, 0, 0, 0), **kwargs):
    if "font" in kwargs:
        font = kwargs['font']
    else:
        font = ImageFont.truetype(r".\resources\fonts\TIMES.TTF", size = 26)
    page_size = width_height(content, font=font, expand_pixel=expand_pixel)
    if isinstance(bg, (tuple, list)):
        color = bg
        canvas_img = Image.new("RGB", size = page_size, color=color)
    elif os.path.isfile(bg):
        canvas_img = Image.fromarray(random_crop(Image.open(bg).convert("RGB"), width=page_size[0], height=page_size[1]))
        
    canvas = ImageDraw.Draw(canvas_img)
    canvas.font = font
    word, bbox = draw_box_word(content, canvas, expand_pixel, **kwargs)
    if draw_bbox:
        draw_bboxes([bbox], ImageDraw.Draw(canvas))
    return canvas._image, bbox, word