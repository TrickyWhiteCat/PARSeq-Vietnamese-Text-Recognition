import resources
import parser
from random import choice, choices, randint, random
from PIL import ImageFont
import numpy as np
from uuid import uuid4 # Import for convienience

def random_margin(margin_min, margin_max):
    margin_left = randint(margin_min, margin_max)
    margin_right = randint(margin_min, margin_max)
    margin_up = randint(margin_min, margin_max)
    margin_down = randint(margin_min, margin_max)
    return margin_left, margin_right, margin_up, margin_down

def roll(chance: float):
    if not isinstance(chance, float):
        raise TypeError("`chance` must be float.")
    return bool(random() < chance)

def random_bw(img, chance = 0.4):
    if random() < chance:
        return img.convert("L").convert("RGB")
    return img.convert("RGB")

def random_font(fonts = resources.FONTPATHS, fontsize = -1):
    global idx
    if fontsize < 0:
        fontsize = random_fontsize()
    fontpath = choice(fonts)
    return ImageFont.truetype(str(fontpath), size = fontsize)

def random_color(channels = 3):
    return tuple([randint(0, 255) for _ in range(channels)])

def random_crop(img, width, height, scale = (0.8, 1.0)):
    from albumentations.augmentations.crops import RandomResizedCrop
    return RandomResizedCrop(size = (height, width), scale=scale)(image=np.array(img))['image']

def random_colors(length = 1, same_chance = 0.3):
    same = random() < same_chance
    if same:
        color = random_color()
        colors = [color] * length
    else:
        colors = [color() for _ in range(length)]
    return colors

def random_background(backgrounds = resources.BACKGROUND_PATHS, blank_chance = 0.3):
    if random() < blank_chance:
        return random_color()
    return choice(backgrounds)

def random_fontsize(minsize = 100, maxsize = 100):
    return randint(minsize, maxsize)

def random_word_kind(options = ["w", "p", "e", "d", "D", "m"], weights = [8, 1, 1, 1, 1, 5]):
    return f"{choices(options, weights = weights, k=1)[0]}"

def random_content():
    kind = random_word_kind()
    if kind == 'w':
        return random_word()
    word = parser.get_format(kind)
    return word

def insert_str(word, pos, ins):
    chars = list(word)
    chars.insert(pos, ins)
    return ''.join(chars)

def random_word(special_chance = 0.2, quote_chance = 0.3):
    word = choice(resources.WORDS)
    while True:
        if random() > special_chance:
            break
        else:
            word = insert_str(word, randint(0, len(word)), parser.get_special())
    if random() < quote_chance:
        word = insert_str(word, 0, choice(resources.QUOTES))
        if random() < quote_chance:
            word = insert_str(word, -1, choice(resources.QUOTES))
    return word