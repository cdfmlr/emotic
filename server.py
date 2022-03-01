import traceback

from aiohttp import web

import cv2
import numpy as np
import os

import torch
from torchvision import transforms

from emopic.emotic.yolo_inference import get_bbox
from emopic.emotic.yolo_utils import prepare_yolo
from emotic import Emotic

from inference import inference_emotic, infer

# region init

# Magic constants: copy from main.py

cat = ['Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion', 'Confidence', 'Disapproval', 'Disconnection', \
       'Disquietment', 'Doubt/Confusion', 'Embarrassment', 'Engagement', 'Esteem', 'Excitement', 'Fatigue', 'Fear',
       'Happiness',
       'Pain', 'Peace', 'Pleasure', 'Sadness', 'Sensitivity', 'Suffering', 'Surprise', 'Sympathy', 'Yearning']
cat2ind = {}
ind2cat = {}
for idx, emotion in enumerate(cat):
    cat2ind[emotion] = idx
    ind2cat[idx] = emotion

vad = ['Valence', 'Arousal', 'Dominance']
ind2vad = {}
for idx, continuous in enumerate(vad):
    ind2vad[idx] = continuous

context_mean = [0.4690646, 0.4407227, 0.40508908]
context_std = [0.2514227, 0.24312855, 0.24266963]
body_mean = [0.43832874, 0.3964344, 0.3706214]
body_std = [0.24784276, 0.23621225, 0.2323653]
context_norm = [context_mean, context_std]
body_norm = [body_mean, body_std]

# configs

gpu = 0
result_path = '../results'
model_path = '../models'

# init

device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

yolo = prepare_yolo(model_path)
yolo = yolo.to(device)
yolo.eval()

thresholds = torch.FloatTensor(np.load(os.path.join(result_path, 'val_thresholds.npy'))).to(device)

model_context = torch.load(os.path.join(model_path, 'model_context1.pth')).to(device)
model_body = torch.load(os.path.join(model_path, 'model_body1.pth')).to(device)
emotic_model = torch.load(os.path.join(model_path, 'model_emotic1.pth')).to(device)
model_context.eval()
model_body.eval()
emotic_model.eval()

models = [model_context, model_body, emotic_model]


# endregion init

# 给人用的接口
def yolo_emotic_infer(image_file, verbose=False):
    """Infer on an image_file to obtain bounding boxes of persons in the images using yolo model.
    And then get the emotions using emotic models.

    :param image_file: image file to do inference: a path str or IO
    :param verbose: print the result
    :return: infer result: a list of bbox of a person in the image,
             the categorical Emotions list and continuous emotion dimensions.
             [{'bbox': [x1, y1, x2, y2], 'cat': ['Sadness', ...], 'cont': [5.8, 7.1, 2.1]}, ...]
    """

    result = []

    try:
        if isinstance(image_file, str):
            if not os.path.exists(image_file):
                raise ValueError('path not exist:', image_file)
            img = cv2.imread(image_file)
        else:  # try to read it
            img_array = np.asarray(bytearray(image_file.read()))
            img = cv2.imdecode(img_array, None)

        image_context = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bbox_yolo = get_bbox(yolo, device, image_context)
        for pred_bbox in bbox_yolo:
            pred_cat, pred_cont = infer(context_norm, body_norm, ind2cat, ind2vad, device, thresholds, models,
                                        image_context=image_context, bbox=pred_bbox, to_print=False)
            result.append({
                'bbox': pred_bbox.tolist(),
                'cat': pred_cat,
                'cont': pred_cont.tolist()})
        if verbose:
            print(f'inference: {image_file=}, {result=}')
    except Exception as e:
        print('Exception for image ', image_file, ':', e)
        traceback.print_exc()
    finally:
        return result


# region webserver

# 其实我想做函数计算，直接上云
# 但这个依赖包+预训练模型加起来太大了，aliyun 似乎不行
# 就先手写个服务吧。

# curl -F "img=@/Users/c/Desktop/IMG_9095.png" http://localhost:8080/infer
async def handle(request):
    data = await request.post()
    img = data['img'].file

    result = yolo_emotic_infer(img)
    return web.json_response(result)


app = web.Application()
app.add_routes([web.post('/infer', handle)])

# endregion webserver

if __name__ == '__main__':
    # Call yolo_emotic_infer to do jobs
    # r = yolo_emotic_infer('/Users/c/Projects/murecom/verse-1/emopic/experiment/elated.jpg')
    # or:
    # with open('/Users/c/Desktop/IMG_9095.png', 'rb') as f:
    #     r = yolo_emotic_infer(f, verbose=True)
    # print(r)

    web.run_app(app)
