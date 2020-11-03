import numpy as np

def concat_tiles(tiles, patch_size, tile_sz=6):
    ret = np.zeros((patch_size*tile_sz, patch_size*tile_sz, 3), dtype=np.uint8)
    for i, tile in enumerate(tiles):
        h, w = i//tile_sz, i%tile_sz
        ret[h*patch_size:(h+1)*patch_size, w*patch_size: (w+1)*patch_size] = tile
    return ret