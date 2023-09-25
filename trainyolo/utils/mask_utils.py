import numpy as np
from trainyolo.utils.rle import rle_to_mask
from PIL import Image
from trainyolo.utils.colormap import get_colormap

def annotations_to_image(annotations, size):

    w, h = size
    inst = np.zeros((h, w), dtype=np.uint16)
    cls = np.zeros((h, w), dtype=np.uint8)

    for i, ann in enumerate(annotations):
        mask = rle_to_mask(ann['segmentation'])
        
        x, y, w, h = ann['bbox']

        # hack for overlapping masks
        inst_region = inst[y:y+h,x:x+w]
        mask = mask * (inst_region == 0)

        inst[y:y+h,x:x+w] += mask * (i+1) 
        cls[y:y+h,x:x+w] += mask * ann["category_id"]

    cls = Image.fromarray(cls, mode='P')
    cls.putpalette(get_colormap())

    inst = Image.fromarray(inst, mode='I;16')

    return inst, cls