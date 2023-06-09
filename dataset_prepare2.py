import os

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


# convert string to integer
def atoi(s):
    n = 0
    for i in s:
        n = n * 10 + ord(i) - ord("0")
    return n

# membuat folder
outer_names = ['test', 'train']
inner_names = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
os.makedirs('data', exist_ok=True)
for outer_name in outer_names:
    os.makedirs(os.path.join('data', outer_name), exist_ok=True)
    for inner_name in inner_names:
        os.makedirs(os.path.join('data', outer_name, inner_name), exist_ok=True)

# menginisialisasi variabel hitung
count = {emotion: 0 for emotion in inner_names}
count_test = {emotion: 0 for emotion in inner_names}

df = pd.read_csv('fer2013.csv')
mat = np.zeros((48, 48), dtype=np.uint8)
print("Saving images...")

# membaca file CSV baris per baris
for i in tqdm(range(len(df))):
    txt = df['pixels'][i]
    words = txt.split()

    # ukuran gambar adalah 48x48
    for j in range(2304):
        xind = j // 48
        yind = j % 48
        mat[xind][yind] = atoi(words[j])

    img = Image.fromarray(mat)

    # train
    if i < 28709:
        emotion = inner_names[df['emotion'][i]]
        img.save(f"data/train/{emotion}/im{count[emotion]}.png")
        count[emotion] += 1
    # test
    else:
        emotion = inner_names[df['emotion'][i]]
        img.save(f"data/test/{emotion}/im{count_test[emotion]}.png")
        count_test[emotion] += 1

print("Done!")
