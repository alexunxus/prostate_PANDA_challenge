import numpy as np
import random # shuffling

def concat_tiles(tiles, patch_size, tile_sz=6):
    '''concat tiles into (3, tile_sz*patch_size, tile_size*patch_size) image'''
    ret = np.ones((patch_size*tile_sz, patch_size*tile_sz, 3), dtype=np.uint8)*255
    if len(tiles) == 0:
        return ret
    for i in range(tile_sz**2):
        tile = tiles[i%len(tiles)]
        h, w = i//tile_sz, i%tile_sz
        ret[h*patch_size:h*patch_size + tile.shape[0], w*patch_size: w*patch_size+tile.shape[1]] = tile
    return ret

def shuffle_two_arrays(imgs, labels):
    zip_obj = list(zip(imgs, labels))
    random.shuffle(zip_obj)
    return zip(*zip_obj)

def binary_search(li, x, lo, hi):
    mid = (lo+hi)//2
    if li[mid][0] == x:
        return li[mid][1]
    if mid == hi:
        raise ValueError(f"{x} is not in list!")
    if li[mid][0] > x:
        return binary_search(li, x, lo, mid)
    else:
        return binary_search(li, x, mid+1, hi)