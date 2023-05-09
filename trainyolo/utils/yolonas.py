from trainyolo.client import MLModel

def upload_yolonas_run(project, checkpoint_path, model_name, imgsz=640, conf=0.5, iou=0.45, metrics={}):

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

    # add version
    print(f'adding checkpoint: {checkpoint_path} to project ...')
    model.add_version(
        checkpoint_path,
        categories=project.categories,
        architecture='yolonas',
        params={
            'model': model_name,
            'imgsz': imgsz,
            'conf': conf,
            'iou': iou
        },
        metrics=metrics
    )
