from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from starlette.applications import Starlette
from starlette.responses import HTMLResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
from pathlib import Path
import uvicorn, aiohttp, asyncio
import base64, sys, numpy as np
import matplotlib.pyplot as plt
import numpy as np
import shutil
from os import listdir

from bs4 import BeautifulSoup as BS
import bs4

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import time
import os
import PIL.Image as Image
from IPython.display import display

path = Path(__file__).parent
#model_file_url = 'https://dl.dropboxusercontent.com/s/egjbcj13vdaba1q/cpu_model.h5?dl=0'
#model_file_name = 'model'

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

MODEL_PATH = path/'models/cpu_model_state_dict.h5'
IMG_FILE_SRC = '/tmp/saved_image.png'

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_model():
    #UNCOMMENT HERE FOR CUSTOM TRAINED MODEL
   # await download_file(model_file_url, MODEL_PATH)
    # Model class must be defined somewhere
    device = torch.device('cpu')
    model = models.resnet34(pretrained=True)
    num_ftrs = model.fc.in_features
    # replace the last fc layer with an untrained one (requires grad by default)
    model.fc = nn.Linear(num_ftrs, 5)
    # model_ft = model_ft.to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    #model = load_model(MODEL_PATH) # Load your Custom trained model
    #model._make_predict_function()
    #model = ResNet50(weights='imagenet') # COMMENT, IF you have Custom trained model
    print("setup_model complete")
    return model

# Asynchronous Steps
loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_model())]
model = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route("/upload", methods=["POST"])
async def upload(request):
    data = await request.form()
    img_bytes = data["img"]
    bytes = base64.b64decode(img_bytes)
    with open(IMG_FILE_SRC, 'wb') as f: f.write(bytes)
    return model_predict(IMG_FILE_SRC, model)

def model_predict(img_path, model):
    device = torch.device("cpu")
    def find_classes(dir):
        cartype = 'Other Body types'
        if(predicted.item()==0):
            cartype = 'Convertible.html'
        elif(predicted.item()==1):
            cartype='Coupe.html'
        elif(predicted.item()==2):
            cartype='Hatchback.html'
        elif(predicted.item()==3):
            cartype='Sedan.html'
        else:
            cartype='SUV.html'
        return cartype
    

    # model = torch.load('../Car/model_rsnet50_cpu.h5', map_location=lambda storage, location: storage)

    dataset_dir = "../Car/car_data_new/"
    model.eval()
    # transforms for the input image
    loader = transforms.Compose([transforms.Resize((400, 400)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    #img_loc = "../Car/Prediction_folder/"
    #file_name="1445460302-mini-convertible.jpg"
    #dest_folder = ‘/tmp’/'static'/'vendor'/'result_files'
    #for file in  listdir(dest_folder):
     #   if file.lower().endswith(('.png', '.jpg', '.jpeg','jfif')):
    #        os.remove(file)
    #print("Copying File")
   # shutil.copy(img_path, ‘/tmp’/'static'/'vendor'/'result_files')
   # print("img path: ",img_path)
    
    #print("Copied File")
    #for file in  listdir(dest_folder):
    #    if file.lower().endswith(('.png', '.jpg', '.jpeg','jfif')):
    #        print(file)
    #        os.rename(file, ‘/tmp’/'static'/'vendor'/'result_files'/'image.jpg')
        
            
    image = Image.open(img_path)
    image = loader(image).float()
    image = torch.autograd.Variable(image, requires_grad=True)
    image = image.unsqueeze(0)
    output = model(image)
    _label, predicted = torch.max(output.data, 1)
    # get the class name of the prediction
    
    
    print("==============================================================================================")
    #display(Image.open(img_loc+file_name))
    print(find_classes(predicted.item()))
    file=find_classes(predicted.item())
    result_html2=path/'static'/file
    result_html = str(result_html2.open().read())
    return  HTMLResponse(result_html)




@app.route("/")
def form(request):
    index_html = path/'static'/'index.html'
    return HTMLResponse(index_html.open().read())

if __name__ == "__main__":
    if "serve" in sys.argv: uvicorn.run(app, host="0.0.0.0", port=8080)
