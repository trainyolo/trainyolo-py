import os
import glob
from trainyolo.client import MLModel
import yaml
import re
import numpy as np
import cv2

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

def mask_to_polygons(mask, offset_x=0, offset_y=0, norm_x=1, norm_y=1):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for polygon in contours:
        polygon = ((polygon.squeeze() + np.array([offset_x, offset_y])) / np.array([norm_x, norm_y])).ravel().tolist()
        polygons.append(polygon)

    return polygons

def annotations_to_yolo_polygons(annotations, im_w, im_h):
    output = []
    for ann in annotations:
        cl, bbox, rle = ann['category_id'], ann['bbox'], ann['segmentation']
        mask = rle_to_mask(rle)
        polygons = mask_to_polygons(mask, bbox[0], bbox[1], im_w, im_h)
        output.append([cl-1, polygons[0]])
    return output

def upload_yolov8_run(project, mode='detect', run_location=None, run=None, weights='best.pt', conf=0.25, iou=0.45):
    run_location = f'./runs/{mode}' or run_location

    # exp path
    if run is None:
        # get latest exp
        exp_paths = glob.glob(os.path.join(run_location, 'train*'))
        # order paths (natural ordering so a bit tricky)
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        exp_path = sorted(exp_paths, key=alphanum_key)[-1]
    else:
        exp_path = os.path.join(run_location, run)

    # get result of best/last model
    csv_file = os.path.join(exp_path, 'results.csv')
    with open(csv_file, 'r') as f:
        lines = [[val.strip() for val in r.split(",")] for r in f.readlines()]
        headers, csv_list = lines[0], lines[1:]

        if mode == 'detect':
            results = [{'precision': float(item[4]), 'recall':float(item[5]), 'map50':float(item[6]), 'map':float(item[7])} for item in csv_list]
        else: #segment
            results = [{
                'precision': float(item[5]), 
                'recall':float(item[6]), 
                'map50':float(item[7]), 
                'map':float(item[8]),
                'precision_mask': float(item[9]), 
                'recall_mask':float(item[10]), 
                'map50_mask':float(item[11]), 
                'map_mask':float(item[12]),
                } for item in csv_list]

    if weights == 'best.pt':
        # find best result
        best_result, best_fi = None, -1
        for result in results:
            if mode == 'detect':
                fi = 0.1*result['map50'] + 0.9*result['map']
            else: #segment
                fi = 0.1*result['map50'] + 0.9*result['map'] + 0.1*result['map50_mask'] + 0.9*result['map_mask']
            if fi > best_fi:
                best_fi = fi
                best_result = result
        result = best_result
    else:
        # take last result
        result = results[-1]

    # get categories
    opt_file = os.path.join(exp_path, 'args.yaml')
    with open(opt_file, 'r') as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)
    
    dataset_file = opt['data']
    with open(dataset_file, 'r') as f:
        dataset = yaml.load(f, Loader=yaml.FullLoader)
    categories = [{'id': id+1, 'name': name} for id, name in dataset['names'].items()]
    
    # create model
    if project.model is None:
        model = MLModel.create(
            project.client,
            f'{project.name}',
            description='',
            type='BBOX' if mode == 'detect' else 'INSTANCE_SEGMENTATION',
            public = project.public,
            project=project.uuid
        )
    else:
        model = project.model

    print(f'adding weights: {os.path.join(exp_path, "weights", weights)} to project ...')
    model.add_version(
        os.path.join(exp_path, 'weights', weights),
        categories=categories,
        architecture='yolov8' if mode == 'detect' else 'yolov8-seg',
        params={
            'model': opt['model'],
            'imgsz': opt['imgsz'],
            'conf': result.get('conf', conf),
            'iou': iou
        },
        metrics={
            'map50': round(result['map50'], 3),
            'map': round(result['map'], 3),
            'precision': round(result['precision'], 3),
            'recall': round(result['recall'], 3)
        } if mode == 'detect' else {
            'map50': round(result['map50'], 3),
            'map': round(result['map'], 3),
            'precision': round(result['precision'], 3),
            'recall': round(result['recall'], 3),
            'map50_mask': round(result['map50_mask'], 3),
            'map_mask': round(result['map_mask'], 3),
            'precision_mask': round(result['precision_mask'], 3),
            'recall_mask': round(result['recall_mask'], 3)
        }
    )