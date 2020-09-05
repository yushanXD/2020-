from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm

import argparse
import fire
import numpy as np
import os
    
    
def _extract_audio_features(file_mel, file_out):
    os.system('./lpcnet_demo -synthesis %s %s' % (file_mel, file_out))


def main(dir_mel, dir_out, file_list=None, num_workers=4):
    os.makedirs(dir_out, exist_ok=True)
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    if file_list is not None:
        with open(file_list) as fin:
            fids = [x.strip() for x in fin.readlines()]
    else:
        fids = sorted([x.split('.')[0] for x in os.listdir(dir_mel)])
    for fid in fids:
        file_mel = os.path.join(dir_mel, fid + '.mel')
        file_out = os.path.join(dir_out, fid + '.wav')
        futures.append(executor.submit(
            partial(_extract_audio_features, file_mel, file_out)))

    results = [future.result() for future in tqdm(futures)]


if __name__ == "__main__":
    fire.Fire(main)
