import os
from pathlib import Path
from trainyolo.client import Client, Project
from trainyolo.utils.yolov5 import upload_yolov5_run
from trainyolo.utils.yolov8 import upload_yolov8_run
import glob
import sys

def _save_apikey(apikey):
    config_path = Path().home() / '.trainyolo' / 'credentials'
    config_path.parent.mkdir(exist_ok=True)

    with open(config_path, 'w') as f:
        f.write(f"apikey={apikey}\n")

def _load_apikey():
    config_path = Path().home() / '.trainyolo' / 'credentials'
    
    try:
        with open(config_path, 'r') as f:
            apikey = f.readline().split('=',1)[1].strip()
            return apikey
    except:
        print('No API Key detected. First authenticate.')
        sys.exit(1)

def _get_client():
    apikey = _load_apikey()
    client = Client(apikey)
    return client

def authenticate(apikey):
    try:
        client = Client(apikey)
        _save_apikey(apikey)
    except:
        print('API Key is invalid.')
        sys.exit(1)

def create_project(name, annotation_type, categories):
    client = _get_client()
    try:
        categories = [cat.strip() for cat in categories.split(',')]
        Project.create(client, name, categories=categories, annotation_type=annotation_type)
    except Exception as e:
        print(e)
        sys.exit(1)

def pull_project(name, path, format):
    client = _get_client() 
    try:   
        project = Project.get_by_name(client, name)
        project.pull(location=path, format=format)
    except Exception as e:
        print(e)
        sys.exit(1)

def push_to_project(name, path):
    client = _get_client()
    try:
        project = Project.get_by_name(client, name)
        
        image_files = []
        image_types = (
            '*.png', '*.PNG', 
            '*.jpg','*.JPG', 
            '*.jpeg', '*.JPEG', 
            '*.tiff', '*.TIFF', 
            '*.tif', '*.TIF'
        )
        for t in image_types:
            image_files.extend(glob.glob(os.path.join(path, t)))
        image_files.sort()

        project.push(image_files)

    except Exception as e:
        print(e)
        sys.exit(1)

def add_model(name, type, run_location, run, conf, iou):
    client = _get_client()
    try:
        project = Project.get_by_name(client, name)
        if type == 'yolov5':
            upload_yolov5_run(project, run_location=run_location, run=run, conf=conf, iou=iou)
        elif type == 'yolov8':
            upload_yolov8_run(project, mode='detect', run_location=run_location, run=run, conf=conf, iou=iou)
        elif type == 'yolov8-seg':
            upload_yolov8_run(project, mode='segment', run_location=run_location, run=run, conf=conf, iou=iou)
        elif type == 'yolov8-pose':
            upload_yolov8_run(project, mode='pose', run_location=run_location, run=run, conf=conf, iou=iou)
        else:
            print(f'Type "{type}" is curently not supported. Options are: yolov5|yolov8|yolov8-seg|yolov8-pose')
            sys.exit(1)
    except Exception as e:
        print(e)
        sys.exit(1)