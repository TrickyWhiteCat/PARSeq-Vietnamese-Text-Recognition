from config import *

import os
import shutil
from pathlib import Path
# from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool, Process

from draw import create_new
import randomize
from degrade import apply_transform

def generate():
    # BW chance
    if randomize.roll(0.4):
        img, bbox, word = create_new(bg=(255, 255, 255),
                                  content=randomize.random_content(),
                                  expand_pixel = randomize.random_margin(0, 50),
                                  font = randomize.random_font(),
                                  draw_bbox=False,
                                  text_color = randomize.random_color())
    else:
        img, bbox, word = create_new(bg=randomize.random_background(),
                                      content=randomize.random_content(),
                                      expand_pixel = randomize.random_margin(0, 50),
                                      font = randomize.random_font(),
                                      draw_bbox=False,
                                      text_color = randomize.random_color())
    degraded = apply_transform(img)
    scale_ratio = 0.3
    width, height = degraded.size
    new_width, new_height = int(width * scale_ratio), int(height * scale_ratio)
    
    downscaled = degraded.resize(size = [new_width, new_height])
    return downscaled, word

def worker(*args, **kwargs):
    try:
        img, word = generate()
        fileid = randomize.uuid4().hex
        filepath = IMAGES_DIR/f"{fileid}.png"

        img.save(filepath)
        with open(LABEL_DIR/f"{os.getpid()}.txt", 'a', encoding='utf-8') as f: # 1 label file for each worker to avoid file lock
            f.write(f"{filepath}\t{word}\n")
    except Exception as e: # FIXME: Known error: `ValueError: operands could not be broadcast together with shapes (0,) (4,)`
        raise e
            
def reporter(report_interval = 0.3, *args, **kwargs):
    import time
    import datetime
    start = time.time()
    longest_msg_length = 0
    while True:
        count = len(os.listdir(IMAGES_DIR))
        if count:
            run_for = (time.time() - start)
            sec_per_img = run_for / count
            projected_time = sec_per_img * NUM_FILES
            time_left = int(projected_time - run_for)
            msg = f"Generated {count}/{NUM_FILES}. Generation speed: {1/sec_per_img:.2f} imgs/s. Expect finish after {str(datetime.timedelta(seconds=time_left))}."
            
            msg_length = len(msg)
            if msg_length > longest_msg_length:
                longest_msg_length = msg_length
            paddings = (longest_msg_length - msg_length) * ' '
            msg += paddings
            
            print(msg, end = '\r')
        time.sleep(report_interval)
        if count >= NUM_FILES:
            return
        
def split(label_path = COMBINED_LABEL_FILE, test_size = 0.2):
    import random
    with open(label_path, encoding='utf-8') as f:
        data = f.readlines()
    random.shuffle(data)
    test_length = int(len(data) * test_size)
    test_set = data[:test_length] 
    train_set = data[test_length:]
    with open(TRAIN_LABEL_FILE, mode = 'w', encoding='utf-8') as f:
        f.writelines(train_set)
    with open(TEST_LABEL_FILE, mode = 'w', encoding='utf-8') as f:
        f.writelines(test_set)
    return train_set, test_set
        
def maxsize_in_dir(dir = IMAGES_DIR):
    maxsize = 0
    for item in os.listdir(dir):
        path = dir/item
        if os.path.isfile(path):
            filesize = os.path.getsize(path)
            if filesize > maxsize:
                maxsize = filesize
        else: # Dir
            dirsize = maxsize_in_dir(path)
            if dirsize > maxsize:
                maxsize = dirsize
    return maxsize

def create_lmdb(gtFile: str, outputPath: str, mapSize, inputPath = '.'):
    import subprocess
    subprocess.run(f"python ./create_lmdb_dataset.py --inputPath {inputPath} --gtFile {gtFile} --outputPath {outputPath} --mapSize {mapSize} --checkValid True", shell=True)
     
if __name__ == "__main__":    
    # shutil.rmtree(GENERATED_DIR, ignore_errors=True)
    # os.makedirs(GENERATED_DIR, exist_ok=True)
    # os.makedirs(IMAGES_DIR, exist_ok=True)
    # os.makedirs(LABEL_DIR, exist_ok=True)
    # iterator = range(NUM_FILES)
    
    # report = Process(target=reporter)
    # report.start()
    # with Pool(NUM_PROCESSORS) as p:
    #     r = list(p.map(worker, iterator))
    
    # # Combine label files into 1
    # label_files = os.listdir(LABEL_DIR)
    # for label_file in label_files:
    #     label_path = LABEL_DIR / label_file
    #     with open(label_path, mode = 'r', encoding='utf-8') as f:
    #         partial_labels = f.read()
    #     with open(COMBINED_LABEL_FILE, mode = 'a', encoding='utf-8') as f:
    #         f.write(partial_labels)
            
    # train_set, test_set = split(test_size=TEST_SPLIT_SIZE)
    with open(TRAIN_LABEL_FILE, mode='r', encoding='utf-8') as f:
        train_set = f.readlines()
    train_size = sum([os.path.getsize(fp.split('\t')[0]) for fp in train_set])
    # test_size = sum([os.path.getsize(fp.split('\t')[0]) for fp in test_set])
    create_lmdb(TRAIN_LABEL_FILE, TRAIN_LMDB_PATH, int(train_size * (1 + LMDB_REDUNDANCY_RATIO)))
    # create_lmdb(TEST_LABEL_FILE, TEST_LMDB_PATH, int(test_size * (1 + LMDB_REDUNDANCY_RATIO)))