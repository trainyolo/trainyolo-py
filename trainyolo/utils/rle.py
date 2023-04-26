from trainyolo.utils.rle_cython import compute_rle
import numpy as np

def mask_to_rle(mask):
    counts = compute_rle(mask.flatten())
    size = [mask.shape[0], mask.shape[1]]
    return {"counts": counts, "size": size}    

def rle_to_mask(rle):
    (h, w), counts = rle['size'], rle['counts']

    mask = np.zeros(w*h, dtype=np.uint8)

    index = 0
    zeros = True
    for count in counts:
        if not zeros:
            mask[index : index + count] = 255
        index+=count
        zeros = not zeros

    mask = np.reshape(mask, [h, w])
    
    return mask