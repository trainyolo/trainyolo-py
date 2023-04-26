import os
import glob
from trainyolo.client import MLModel
import yaml
import re
from trainyolo.utils.ocr import read_f1_conf

def format_boxes(boxes):
    annotations = []
    for box in boxes:
        x1, y1, x2, y2, _, cl = box.tolist()
        w, h = x2 - x1, y2 - y1
        annotations.append({
            'bbox': [x1, y1, w, h],
            'area': w * h,
            'category_id': int(cl) + 1
        })
    return annotations

def upload_yolov5_run(project, run_location=None, run=None, weights='best.pt', conf=None, iou=0.45):
    run_location = run_location or './runs'
    run_location = os.path.join(run_location, 'train')
    # exp path
    if run is None:
        # get latest exp
        exp_paths = glob.glob(os.path.join(run_location, '*'))
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
        results = [{'precision': float(item[4]), 'recall':float(item[5]), 'map50':float(item[6]), 'map':float(item[7])} for item in csv_list]

    if weights == 'best.pt':
        # find best result
        best_result, best_fi = None, -1
        for result in results:
            fi = 0.1*result['map50'] + 0.9*result['map']
            if fi > best_fi:
                best_fi = fi
                best_result = result
        result = best_result
    else:
        # take last result
        result = results[-1]

    # get categories
    opt_file = os.path.join(exp_path, 'opt.yaml')
    with open(opt_file, 'r') as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)
    
    dataset_file = opt['data']
    with open(dataset_file, 'r') as f:
        dataset = yaml.load(f, Loader=yaml.FullLoader)
    categories = [{'id': id+1, 'name': name} for id, name in dataset['names'].items()]

    # read conf from f1 curve
    if conf is None:
        print('Reading best conf from F1_curve')
        f1_curve = os.path.join(exp_path, 'F1_curve.png')
        if os.path.exists(f1_curve):
            conf = read_f1_conf(project.client, f1_curve)
            if conf > 0:
                print(f'Using conf={conf}, which maximizes f1 score.')
                conf = conf
            else:
                print("Something went wrong while reading f1 curve, defaulting to conf=0.5")
                conf = 0.5
        else:
            print("Unable to find f1 curve, defaulting to conf=0.5")
            conf = 0.5
    
    # create model
    if project.model is None:
        model = MLModel.create(
            project.client,
            f'{project.name}',
            description='',
            type='BBOX',
            public = project.public,
            project=project.uuid
        )
    else:
        model = project.model

    print('adding version to project ...')
    model.add_version(
        os.path.join(exp_path, 'weights', weights),
        categories=categories,
        architecture='yolov5',
        params={
            'model': opt['weights'],
            'imgsz': opt['imgsz'],
            'conf': conf,
            'iou': iou
        },
        metrics={
            'map50': round(result['map50'], 3),
            'map': round(result['map'], 3),
            'precision': round(result['precision'], 3),
            'recall': round(result['recall'], 3)
        }
    )