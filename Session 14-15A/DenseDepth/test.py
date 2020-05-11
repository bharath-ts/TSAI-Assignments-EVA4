
import os
import glob
import argparse
import matplotlib
from tqdm import tqdm
# Keras / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from layers import BilinearUpSampling2D
from utils import predict, load_images, display_images, save_images
from matplotlib import pyplot as plt
import numpy as np
import skimage
from skimage.transform import resize
from skimage import io, img_as_ubyte
import json 
from PIL import Image
import zipfile


# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--model', default='nyu.h5', type=str, help='Trained Keras model file.')
parser.add_argument('--input', default='examples/*.png', type=str, help='Input filename or folder.')
parser.add_argument('--out_path',  type=str, help='output folder.')

# parser.add_argument('--image_list', nargs="+", help='Input filename or folder.')

args = parser.parse_args()

# Custom object needed for inference and training
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

print('Loading model...')
# Load model into GPU / CPU
model = load_model(args.model, custom_objects=custom_objects, compile=False)
print('\nModel loaded ({0}).'.format(args.model))

# Input images
with open('/content/drive/My Drive/Session 14-15A/DenseDepth/img_list_new_backup.json', 'r') as infile:
    input_files_list = json.load(infile)

print("Len of input_files_list", len(input_files_list))
lim = int((len(input_files_list))/4)

input_files_list = input_files_list[0:lim]
print("new len of input_files_list", len(input_files_list))

output_path = '/content/drive/My Drive/Session 14-15A/Baskball_Players_DepthEstimation/2/'
# input_path =  '/content/drive/My Drive/Session 14-15A/Baskball_Players_DepthEstimation/FG_BG/'


st_batch = 13500

for batch in tqdm(range(13500,len(input_files_list),100)):
    step=100
    end_batch=st_batch + step
    file_name_list = []  
    loaded_images = []

    with zipfile.ZipFile('/content/drive/My Drive/Session 14-15A/depth_fg_bg_new_reduced.zip','r') as zip:
      inflist = zip.namelist()
      for f in inflist[st_batch:end_batch]:
        file1 = zip.open(f)
        img = x = np.clip(np.asarray(Image.open(file1), dtype=float) / 255, 0, 1)
        loaded_images.append(img)
        file_name_list.append(os.path.splitext(os.path.basename(file1.name))[0])
    
    inputs = np.stack(loaded_images, axis=0)

    del loaded_images, img, file1, inflist, zip

    # print('\nLoaded ({0}) images of size {1}.'.format(inputs.shape[0], inputs.shape[1:]))
    st_batch = batch+step

    # Compute results
    outputs = predict(model, inputs, batch_size=10)

    # print(outputs.shape, inputs.shape)
    
    for ido, output in enumerate(outputs):    
        output_img = Image.fromarray((output[:, :, 0] * 255).astype(np.uint8))
        output_img.save(output_path + file_name_list[ido] + ".jpeg")
    
    del inputs, outputs, output, output_img, ido, file_name_list
    # print("outputs saved")