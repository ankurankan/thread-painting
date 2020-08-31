from itertools import combinations, chain

from skimage.draw import line
from skimage.io import imread
from skimage.transform import rescale
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

THREAD_VALUE = 5

def line_diff(img, orig_img, start, end):
    if (start[0] != end[0]) and (start[1] != end[1]):
        return start, end, np.inf
    img_copy = img.copy()
    rr, cc = line(start[0], start[1], end[0], end[1])
    img_copy[rr, cc] += THREAD_VALUE
    img_copy = img_copy.clip(0, 255)
    diff = np.sum(np.absolute(orig_img - img_copy))
    return start, end, diff

def add_line(orig_img, line_img):
    x_len, y_len = line_img.shape
    lower_edge = [(i, 0) for i in range(x_len)]
    upper_edge = [(i, y_len-1) for i in range(x_len)]
    left_edge = [(0, i) for i in range(y_len)]
    right_edge = [(x_len-1, i) for i in range(y_len)]

    values = Parallel(n_jobs=-1)(delayed(line_diff)(line_img, orig_img, start, end) for start, end in combinations(chain(lower_edge, upper_edge, left_edge, right_edge), 2))
    start, end, diff= min(values, key=lambda t: t[2])
    return line(start[0], start[1], end[0], end[1]), diff

#    least_diff = np.inf
#    best_line = None
#    for start, end in combinations(chain(lower_edge, upper_edge, left_edge, right_edge), 2):
#        if (start[0] != end[0]) and (start[1] != end[1]):
#            # import pdb; pdb.set_trace()
#            orig_copy = line_img.copy()
#            rr, cc = line(start[0], start[1], end[0], end[1])
#            orig_copy[rr, cc] += THREAD_VALUE
#            orig_copy = np.clip(orig_copy, 0, 255)
#            diff = np.sum(np.absolute(orig_img - orig_copy))
#            if diff < least_diff:
#                least_diff = diff
#                best_line = (rr, cc)
#    return best_line, least_diff

def generate_image(orig_img, start_image=None, max_iter=int(1e4)):
    if start_image is None:
        img = np.zeros((orig_img.shape[0], orig_img.shape[1]), dtype='int8')
    else:
        img = start_image
    total = np.sum(orig_img)
    value = np.inf
    curr_diff = np.inf
    iter_no = 0

    pbar = tqdm()
    while (iter_no < max_iter):
        best_line, diff = add_line(orig_img, img)
        if diff < curr_diff:
            curr_diff = diff
            img[best_line[0], best_line[1]] += THREAD_VALUE
            img = np.clip(img, 0, 255)
            iter_no += 1
            pbar.update(1)
            pbar.set_description(desc=f"Total: {np.sum(img)}/{total}", refresh=True)
        else:
            break
    return img

def load_image(filename, scale=1):
    img = imread(filename, as_gray=True)
    return np.array(rescale(img, scale)*256, dtype='int8')
