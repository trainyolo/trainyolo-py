import requests
import os
from urllib.request import urlretrieve
from multiprocessing import Pool
from tqdm import tqdm
from itertools import repeat
import yaml
import time

class OCRResult:
    def __init__(self, client, data):
        self.client = client
        self.data = data

    def update(self):
        data = self.client.get(f'/functions/ocr/{self.uuid}/')
        self.data = data

    @property
    def uuid(self):
        return self.data['uuid']

    @property
    def result(self):
        return self.data['result']
    
    @property
    def processing_status(self):
        return self.data['processing_status']
    
    @property
    def processing_msg(self):
        return self.data['processing_msg']
    
    @classmethod
    def create(cls, client, image):

        asset = Asset.create(client, image)

        payload = {
            'image': asset.uuid
        }
        data = client.post(f'/functions/ocr/', payload)

        return cls(client, data)

    def get_result(self, retry=30):
        r = 0
        while(self.processing_status == 'PENDING'):
            self.update()
            time.sleep(1)
            r+=1
            if r > retry:
                break
        return self.processing_status, self.result

class Asset:

    def __init__(self, client, data):
        self.client = client
        self.data = data

    @property
    def uuid(self):
        return self.data['uuid']

    @property
    def filename(self):
        return self.data['filename']

    @property
    def asset(self):
        return self.data['asset']

    @classmethod
    def create(cls, client, file, type='OTHER'):
        # create asset
        filename = os.path.basename(file)
        payload = {
            'filename': filename,
            'type': type
        }
        data = client.post(f'/assets/', payload=payload)

        # upload file to s3
        url = data['presigned_post_fields']['url']
        fields = data['presigned_post_fields']['fields']
        
        with open(file, 'rb') as f:
            response = requests.post(url, data=fields, files={'file':f})
        
        if response.status_code != 204:
            raise Exception(f"Failed uploading to s3, status_code: {response.status_code}")

        # confirm upload
        data = client.put(f'/assets/{data["uuid"]}/confirm_upload/')

        return cls(client, data)

class MLModelVersion:

    def __init__(self, client, data):
        self.client = client
        self.data = data

    @property
    def version(self):
        return self.data['version']

    @property
    def categories(self):
        return self.data['categories']

    @property
    def architecture(self):
        return self.data['architecture']

    @property
    def weights(self):
        return self.data['weights']

    @property
    def params(self):
        return self.data['params']

    @property
    def metrics(self):
        return self.data['metrics']

    @property
    def created_at(self):
        return self.data['created_at']

    @classmethod
    def create(cls, client, model, weights, categories, architecture, params, metrics={}):

        # create asset
        asset = Asset.create(client, weights)

        # create version
        payload = {
            'categories': categories,
            'architecture': architecture,
            'weights': asset.uuid,
            'params': params,
            'metrics': metrics
        }
        data = client.post(f'/models/{model}/versions/', payload)

        return cls(client, data)

class MLModel:

    def __init__(self, client, data):
        self.client = client
        self.data = data

    @property
    def uuid(self):
        return self.data['uuid']

    @property
    def name(self):
        return self.data['name']

    @property
    def description(self):
        return self.data['description']

    @property
    def type(self):
        return self.data['type']

    @property
    def project(self):
        return Project.get_by_uuid(self.client, self.data['project'])

    @property
    def thumbnail(self):
        return self.data['thumbnail']

    @property
    def public(self):
        return self.data['public']

    @property
    def owner(self):
        return User(self.data['owner'])

    @property
    def latest_version(self):
        if self.data['latest_version']:
            return MLModelVersion(self.client, self.data['latest_version'])
        else:
            return None

    @property
    def created_at(self):
        return self.data['created_at']

    @classmethod
    def create(cls, client, name, description='', type='BBOX', public=False, project=None):
        payload = {
            'name': name,
            'description': description,
            'type': type,
            'public': public,
            'project': project
        }
        data = client.post('/models/', payload)

        return cls(client, data)

    @classmethod
    def get_or_create(cls, client, name, owner=None, **kwargs):
        try:
            model = cls.get_by_name(client, name, owner=owner)
        except:
            model = cls.create(client, name, **kwargs)

        return model

    @classmethod
    def get_by_uuid(cls, client, uuid):
        data = client.get(f'/models/{uuid}/')
        return cls(client, data)

    @classmethod
    def get_by_name(cls, client, name, owner=None):
        owner = owner if owner else client.owner.username
        data = client.get(f'/models/?name={name}&owner={owner}')
        
        if len(data) == 0:
            raise Exception(f'Did not found model "{name}" for owner "{owner}". If this is a shared or public model, please specify the owner')

        return cls(client, data[0])

    def add_version(self, weights, categories, architecture, params, metrics={}):
        return MLModelVersion.create(self.client, self.uuid, weights, categories, architecture, params, metrics=metrics)

class Video:

    def __init__(self, client, data):
        self.client = client
        self.data = data

    @property
    def uuid(self):
        return self.data['uuid']

    @property
    def name(self):
        return self.data['name']

    @property
    def video(self):
        return self.data['video']
    
    @property
    def created_at(self):
        return self.data['created_at']
    
    @classmethod
    def create(cls, client, project, video_file):
        # create asset
        asset = Asset.create(client, video_file)

        # create sample
        payload = {
            'name': asset.filename,
            'video': asset.uuid,
        }
        data = client.post(f'/projects/{project}/videos/', payload)

        return cls(client, data)
    

class Sample:

    def __init__(self, client, data):
        self.client  = client
        self.data = data

    @property
    def uuid(self):
        return self.data['uuid']

    @property
    def name(self):
        return self.data['name']

    @property
    def asset(self):
        return self.data['asset']

    @property
    def label(self):
        return self.data['label']

    @label.setter
    def label(self, annotations):
        payload = {
            'annotations': annotations
        }
        self.data['label'] = self.client.put(f'/samples/{self.uuid}/label/', payload=payload)

    @property
    def prediction(self):
        return self.data['prediction']
    
    @prediction.setter
    def prediction(self, annotations):
        payload = {
            'annotations': annotations
        }
        self.data['prediction'] = self.client.put(f'/samples/{self.uuid}/prediction/', payload=payload)

    @property
    def approved(self):
        return self.data['approved']
    
    @property
    def tags(self):
        return self.data['tags']

    @property
    def split(self):
        return self.data['split']

    @property
    def is_loading(self):
        return self.data['is_loading']

    @property
    def created_at(self):
        return self.data['created_at']
    
    @classmethod
    def create(cls, client, project, image_file, tags={}):
        # create asset
        asset = Asset.create(client, image_file, type='IMAGE')

        # create sample
        payload = {
            'name': asset.filename,
            'asset': asset.uuid,
            'tags': tags
        }
        data = client.post(f'/projects/{project}/samples/', payload)

        return cls(client, data)
    
    @classmethod
    def get_by_uuid(cls, client, uuid):
        data = client.get(f'/samples/{uuid}/')
        return cls(client, data)

    def pull_image(self, location='./'):
        asset_filename = self.asset['filename']
        asset_location = os.path.join(location, 'images', asset_filename)
        if not os.path.exists(asset_location):
            urlretrieve(self.asset['url'], asset_location)

    def pull_label(self, location='./', type='BBOX', format=None):
        if type in ['BBOX','KEYPOINT']:
            if format in ['yolov5', 'yolov8']:
                asset_filename = self.asset['filename']
                label_filename = os.path.splitext(asset_filename)[0] + '.txt'
                label_location = os.path.join(location, 'labels', label_filename)

                im_w, im_h = self.asset['metadata']['size'] 

                with open(label_location, 'w') as f:
                    for l in self.label['annotations']:
                        cl = l['category_id'] - 1
                        x_min, y_min, w, h = l['bbox']
                        x_max = x_min + w
                        y_max = y_min + h

                        x_min = min(max(0, x_min), im_w)
                        y_min = min(max(0, y_min), im_h)
                        x_max = min(max(0, x_max), im_w)
                        y_max = min(max(0, y_max), im_h)

                        xc, yc = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
                        w, h = x_max - x_min, y_max - y_min
                        
                        xc, yc = xc / im_w, yc / im_h
                        w, h = w / im_w, h / im_h

                        f.write(f'{cl} {xc:.5f} {yc:.5f} {w:.5f} {h:.5f}')

                        if type == 'KEYPOINT':
                            keypoints = l['keypoints']
                            for kx, ky, kv in zip(keypoints[0::3], keypoints[1::3], keypoints[2::3]):
                                kx = min(max(0, kx), im_w) / im_w
                                ky = min(max(0, ky), im_h) / im_h
                                f.write(f' {kx:.5f} {ky:.5f} {kv}')

                        f.write('\n')
            else:
                raise Exception(f'Export format {format}" is not supported. Please check the documentation for the formats we support.')
        elif type=='INSTANCE_SEGMENTATION':
            if format in ['yolov5', 'yolov8']:
                from trainyolo.utils.yolov8 import annotations_to_yolo_polygons
                
                asset_filename = self.asset['filename']
                label_filename = os.path.splitext(asset_filename)[0] + '.txt'
                label_location = os.path.join(location, 'labels', label_filename)

                im_w, im_h = self.asset['metadata']['size'] 

                with open(label_location, 'w') as f:
                    annotations = self.label['annotations']
                    yolo_polygons = annotations_to_yolo_polygons(annotations, im_w, im_h)

                    for cl, polygon in yolo_polygons:
                        f.write(f'{cl}')
                        for coord in polygon:
                            f.write(f' {coord:.5f}')
                        f.write('\n')

        else:
            assert False, f'Type {type} is not supported.'

    def pull(self, location='./', type='BBOX', format=None):
        self.pull_image(location=location)
        self.pull_label(location=location, type=type, format=format)

    def delete(self):
        self.client.delete(f'/samples/{self.uuid}/')

class Project:
    
    def __init__(self, client, data):
        self.client = client
        self.data = data

    @property
    def uuid(self):
        return self.data['uuid']

    @property
    def name(self):
        return self.data['name']

    @property
    def description(self):
        return self.data['description']

    @property
    def thumbnail(self):
        return self.data['thumbnail']

    @property
    def categories(self):
        return self.data['categories']

    @property
    def annotation_type(self):
        return self.data['annotation_type']

    @property
    def num_samples(self):
        return self.data['num_samples']

    @property
    def model(self):
        if self.data['model']:
            return MLModel(self.client, self.data['model'])
        else:
            return None

    @property
    def prediction_model(self):
        if self.data['prediction_model']:
            return MLModel(self.client, self.data['prediction_model'])
        else:
            return None

    @property
    def samples(self):
        samples = self.get_samples(filter='')
        return samples

    @property
    def owner(self):
        return User(self.data['owner'])

    @property
    def collaborators(self):
        return [User(data) for data in self.data['collaborators']]

    @property
    def public(self):
        return self.data['public']

    @property
    def created_at(self):
        return self.data['created_at']

    @classmethod
    def create(cls, client, name, categories=['object'], description='', annotation_type='BBOX'):
        if annotation_type in ['BBOX', 'KEYPOINT']:
            category_dict = [{'id': id + 1, 'name': name} for id, name in enumerate(categories)]
        elif annotation_type == 'INSTANCE_SEGMENTATION':
            category_dict = [{'id': id + 1, 'name': name, 'has_instances': True} for id, name in enumerate(categories)]
        else:
            raise Exception(f'Annotation type "{annotation_type}" is not supported. Either use "BBOX" , "KEYPOINT" or "INSTANCE_SEGMENTATION".')

        payload = {
            'name': name,
            'description': description,
            'categories': category_dict,
            'annotation_type': annotation_type
        }
        data = client.post('/projects/', payload)

        return cls(client, data)

    @classmethod
    def get_or_create(cls, client, name, owner=None, *args, **kwargs):
        try:
            project = cls.get_by_name(client, name, owner=owner)
        except:
            project = cls.create(client, name, *args, **kwargs)

        return project

    @classmethod
    def get_by_uuid(cls, client, uuid):
        data = client.get(f'/projects/{uuid}/')
        return cls(client, data)

    @classmethod
    def get_by_name(cls, client, name, owner=None):
        owner = owner if owner else client.owner.username
        data = client.get(f'/projects/?name={name}&owner={owner}')
        
        if len(data) == 0:
            raise Exception(f'Did not found project "{name}" for owner "{owner}". If this is a shared or public project, please specify the owner')

        return cls(client, data[0])

    def pull(self, location='./', filter='LABELED', format=None):

        print('Downloading project')

        # make split
        self.make_split()

        # load samples
        samples = self.get_samples(filter=filter)

        # pull
        project_loc = os.path.join(location, self.name)

        if format in ['yolov5', 'yolov8']:

            image_loc = os.path.join(project_loc, 'images')
            label_loc = os.path.join(project_loc, 'labels')

            os.makedirs(image_loc, exist_ok=True)
            os.makedirs(label_loc, exist_ok=True)

            print('Downloading samples... (this may take a while depending on your dataset size)')
            with Pool(8) as p:
                inputs = zip(samples, repeat(project_loc), repeat(self.annotation_type), repeat('yolov5'))
                r = p.starmap(Sample.pull, tqdm(inputs, total=len(samples)))

            # create train-val txt file
            with open(os.path.join(project_loc, 'train.txt'), 'w') as f_train, open(os.path.join(project_loc, 'val.txt'), 'w') as f_val:
                for s in samples:
                    if s.split == 'TRAIN':
                        f_train.write(f'./images/{s.asset["filename"]}\n')
                    if s.split == 'VAL':
                        f_val.write(f'./images/{s.asset["filename"]}\n')

            # create dataset yaml file
            with open(os.path.join(project_loc, 'dataset.yaml'), 'w') as f:
                yaml_content = {
                    'path': os.path.abspath(project_loc),
                    'train': 'train.txt',
                    'val': 'val.txt',
                    'names': {cat['id']-1:cat['name'] for cat in self.categories}
                }
                if self.annotation_type == 'KEYPOINT':
                    yaml_content['kpt_shape'] = [len(self.categories[0]['keypoints']),3]
                yaml.dump(yaml_content, f, default_flow_style=False)

            return project_loc

        else:
            raise Exception(f'Export format "{format}" is not available. Please check our documentation for the different formats we support.')


    def push(self, image_list):
        with Pool(8) as p:
            inputs = zip(repeat(self), image_list)
            r = p.starmap(Project.add_sample, tqdm(inputs, total=len(image_list)))

    def get_samples(self, filter='LABELED'):

        if filter.lower() == 'labeled':
            query = 'labeled=True'
        elif filter.lower() == 'unlabeled':
            query = 'labeled=False'
        else:
            query = ''

        sample_list = self.client.get(f'/projects/{self.uuid}/sample_list/?{query}')
        
        print('Loading samples...')
        with Pool(8) as p:
            samples = p.starmap(Sample.get_by_uuid, tqdm(zip(repeat(self.client), sample_list), total=len(sample_list)))
        
        return samples

    def add_sample(self, image_file, tags={}):
        return Sample.create(self.client, self.uuid, image_file, tags=tags)

    def make_split(self):
        return self.client.put(f'/projects/{self.uuid}/make_split/')

    def delete(self):
        self.client.delete(f'/projects/{self.uuid}/')

class User:
    def __init__(self, data):
        self.data = data

    @property
    def username(self):
        return self.data['username']

    @property
    def email(self):
        return self.data['email']

class Client:

    def __init__(self, apikey, url="https://api.trainyolo.com"):
        self.url = url
        self.apikey = apikey

    @property
    def owner(self):
        data = self.get('/auth/users/me')
        return User(data)
    
    def get(self, endpoint):

        response = requests.get(self.url + endpoint, headers=self._get_headers())
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Request failed, status_code: {response.status_code}")

    def post(self, endpoint, payload={}, files=None):

        response = requests.post(self.url + endpoint, json=payload, files=files, headers=self._get_headers())

        if response.status_code == 201:
            return response.json()
        else:
            raise Exception(f"Request failed, status_code: {response.status_code} - {response.text}")

    def patch(self, endpoint, payload={}):
        
        response = requests.patch(self.url + endpoint, json=payload, headers = self._get_headers())

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Request failed, status_code: {response.status_code} - {response.text}")

    def put(self, endpoint, payload={}):

        response = requests.put(self.url + endpoint, json=payload, headers = self._get_headers())

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Request failed, status_code: {response.status_code} - {response.text}")

    def delete(self, endpoint):
        response = requests.delete(self.url + endpoint, headers=self._get_headers())
        
        if response.status_code != 204:
            raise Exception(f"Request failed, status_code: {response.status_code}")

    def _get_headers(self):

        # set content type & authorization token
        headers = {
            'Authorization': f"APIKey {self.apikey}"
        }

        return headers

