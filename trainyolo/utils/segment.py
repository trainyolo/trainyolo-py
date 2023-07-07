from trainyolo.client import MLModel
import yaml
import os
from datetime import datetime

def upload_latest_run(project, run_location=None):
    run_location = run_location or './output'

    # get latest run
    runs = os.listdir(os.path.join(run_location, 'train'))
    runs.sort(key=lambda date: datetime.strptime(date, "%m-%d-%Y-%H:%M:%S"), reverse=True)

    run_path = os.path.join(run_location, 'train', runs[0])    

    # read config 
    with open(os.path.join(run_path, 'config.yaml'), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # read results
    with open(os.path.join(run_path, 'best_miou_model.csv'), 'r') as f:
        lines = [[val.strip() for val in r.split(",")] for r in f.readlines()]
        header, values = lines[0], lines[1]

    # create model
    if project.model is None:
        model = MLModel.create(
            project.client,
            f'{project.name}',
            description='',
            type=project.annotation_type,
            public = project.public,
            project=project.uuid
        )
    else:
        model = project.model

    print(f'adding weights: {os.path.join(run_path, "best_miou_model.pt")} to project ...')
    model.add_version(
        os.path.join(run_path, "best_miou_model.pt"),
        categories=project.categories,
        architecture='trainyolo-seg',
        params={
            'model': config['model'],
            'encoder': config['model_encoder'],
            'min_size': config['img_size'],
            'pad': 32,
            'normalize': True,
        },
        metrics={
        'miou': round(float(values[0]), 5),
        'class_iou': {cl:round(float(v),5) for cl, v in zip(header[1:], values[1:])}
        }
    )