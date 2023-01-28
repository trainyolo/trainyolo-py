import os
import glob
from trainyolo.client import MLModel
import yaml
import re

def upload_yolo_run(project, run_location='./runs/train', run=None, weights='best.pt', threshold=0.25, nms_threshold=0.45):
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
        if len(headers) == 15:
            results = [{'precision': float(item[4]), 'recall':float(item[5]), 'map50':float(item[6]), 'map':float(item[7]), 'conf': float(item[-1])} for item in csv_list]
        else:
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
            'img': opt['imgsz'],
            'threshold': result.get('conf', threshold),
            'nms_threshold': nms_threshold
        },
        metrics={
            'map50': round(result['map50'], 3),
            'map': round(result['map'], 3),
            'precision': round(result['precision'], 3),
            'recall': round(result['recall'], 3)
        }
    )