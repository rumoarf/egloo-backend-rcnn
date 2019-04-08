import pika
import redis
import os
from PIL import Image, ImageDraw
from io import BytesIO
import numpy
import skimage
import Config
import matplotlib
import matplotlib.pyplot as plt
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
import EraseAndWrite
import Tesseract
import translate

credential = pika.PlainCredentials(username='cwjwjjwz', password='wCv_GEbhj-KKrkMfQQOCTB-k71Yp3VJM')
parameters = pika.ConnectionParameters(host='moose.rmq.cloudamqp.com', credentials=credential, virtual_host='cwjwjjwz')
# connection = pika.BlockingConnection(pika.URLParameters('amqp://cwjwjjwz:wCv_GEbhj-KKrkMfQQOCTB-k71Yp3VJM@moose.rmq.cloudamqp.com/cwjwjjwz'))
connection = pika.BlockingConnection(parameters=parameters)

r = redis.Redis(host='ec2-3-208-118-12.compute-1.amazonaws.com', password='pbdf7f3407d697d96e664d50f9befb3a113b765c5afdebc85cc54aef563a3dc86', port=29769)

print("connection success")
channel = connection.channel()


def callback(ch, method, properties, body):
    print(properties.headers['token'])
    image = Image.open(BytesIO(body))
    image = image.convert('RGB')
    result = model.detect([numpy.asarray(image)], verbose=1)

    draw = ImageDraw.Draw(image)

    for box in result[0]['rois'].copy():
        box[0], box[1] = box[1], box[0]
        box[2], box[3] = box[3], box[2]
        # draw.rectangle(((box[0], box[1]), (box[2], box[3])), outline='red', width=7)
        croped = image.crop((box[0], box[1], box[2], box[3]))
        data = Tesseract.ocr(croped)
        balloon = EraseAndWrite.erase(image=croped, data=data, color='white')
        # EraseAndWrite.write(image=balloon, data=data)
        image.paste(balloon, (box[0], box[1]))
        
    image.thumbnail([1440, 847], Image.ANTIALIAS)

    imgByteArr = BytesIO()
    image.save(imgByteArr, format='PNG')
    imgByteArr = imgByteArr.getvalue()

    r.set(properties.headers['token'], imgByteArr, ex=300)

channel.basic_consume(callback, 'test', True)


class InferenceConfig(Config.BalloonConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


inference_config = InferenceConfig()
# Recreate the model in inference mode
ROOT_DIR = os.path.abspath("")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
model = modellib.MaskRCNN(
    mode="inference", config=inference_config, model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = model.find_last()

# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

print("starting consuming")
channel.start_consuming()
