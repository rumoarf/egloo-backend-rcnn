import pika
import redis
import json
from PIL import Image
from io import BytesIO

credential = pika.PlainCredentials(username='rumo', password='tardis21')
parameters = pika.ConnectionParameters(host='35.243.66.11', credentials=credential)
# connection = pika.BlockingConnection(pika.URLParameters('amqp://localhost'))
connection = pika.BlockingConnection(parameters=parameters)

r = redis.Redis(host='35.243.66.11', password='test')

print("connection success")
channel = connection.channel()

def callback(ch, method, properties, body):
    print(properties.headers['token'])
    image = Image.open(BytesIO(body))
    image.thumbnail((480, 480), Image.ANTIALIAS)
    r.set(properties.headers['token'], body, ex=30)

channel.basic_consume(callback, queue='test', no_ack=True)
channel.start_consuming()