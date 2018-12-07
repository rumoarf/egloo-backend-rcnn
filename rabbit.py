import pika
from PIL import Image
from io import BytesIO

connection = pika.BlockingConnection(pika.URLParameters('amqp://localhost'))

channel = connection.channel()

def callback(ch, method, properties, body):
    Image.open(BytesIO(body)).show()

channel.basic_consume(callback, queue='test', no_ack=True)
channel.start_consuming()