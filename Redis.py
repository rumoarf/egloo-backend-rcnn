import redis
import io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pytesseract

from PIL import Image

r = redis.Redis('35.221.65.47')

im = open("test.png", 'rb')
# 
# test = int.from_bytes(im.read(), byteorder='big', signed=True)
print(im.read())
# print(type(test))
# r.set("test", im.read())
# r.set("test", int.from_bytes(im.read(), byteorder='big'))

# print(r.get("test"))
