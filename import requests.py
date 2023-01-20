import requests
#file = {'file': open('data/processed/dataset.pt', 'rb')}
import torch
import io

file = {'file': open('data/processed/dataset.pt', 'rb')}

import pickle
import io

bytes_image = pickle.dumps("data/processed/dataset.pt")
stream = io.BytesIO(bytes_image)
files = {"bytes_image": stream}


file = {'file': open("data/processed/dataset.pt", 'rb')}
requests.post(url="http://localhost:8000/predict_model_v4/", files=file)
file
url="http://localhost:8000/predict_model_v4/"
headers = {"Content-Type": "application/json; charset=utf-8"}
    
with open('data/processed/dataset.pt', 'rb') as fobj:
    response = requests.put(url, files={'file': fobj})
response
    
a=requests.post(url="http://localhost:8000/post_1/", params={'id': 'test', 'password': '2'})
a.json()

files = {'my_file': open('README.md', 'rb')}