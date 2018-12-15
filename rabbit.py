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

credential = pika.PlainCredentials(username='rumo', password='tardis21')
parameters = pika.ConnectionParameters(
    host='35.243.66.11', credentials=credential)
# connection = pika.BlockingConnection(pika.URLParameters('amqp://localhost'))
connection = pika.BlockingConnection(parameters=parameters)

r = redis.Redis(host='35.243.66.11', password='test')

print("connection success")
channel = connection.channel()


def callback(ch, method, properties, body):
    print(properties.headers['token'])
    image = Image.open(BytesIO(body))
    result = model.detect([numpy.asarray(image)], verbose=1)

    draw = ImageDraw.Draw(image)

    for box in result[0]['rois'].copy():
        box[0], box[1] = box[1], box[0]
        box[2], box[3] = box[3], box[2]
        draw.rectangle(((box[0], box[1]), (box[2], box[3])), outline='red', width=8)

    image.thumbnail([1024, 720], Image.ANTIALIAS)

    imgByteArr = BytesIO()
    image.save(imgByteArr, format='PNG')
    imgByteArr = imgByteArr.getvalue()
    # r = result[0]

    r.set(properties.headers['token'], imgByteArr, ex=300)
    # result = Image.open(BytesIO(r.get(properties.headers['token'])))
    # result.show()


channel.basic_consume(callback, queue='test', no_ack=True)


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
