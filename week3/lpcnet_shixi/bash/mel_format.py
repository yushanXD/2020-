import numpy as np
import os
import sys
from tqdm import tqdm
def mel_format(mel_npy_dir,out_dir):
    if not os.path.isdir(mel_npy_dir):
        raise Exception('No such directory')
    os.mkdir(out_dir)
    file_list = os.listdir(mel_npy_dir)
    for file in tqdm(file_list):
        path = os.path.join(mel_npy_dir,file)
        mel_npy = np.load(path)
        new_path = os.path.join(out_dir,file.split('.')[0]+'.mel')
        mel_npy.astype(np.float32).tofile(new_path)
    print('done!')


if __name__=='__main__':
    mel_npy_dir = sys.argv[1]
    out_dir = sys.argv[2]
    mel_format(mel_npy_dir,out_dir)
