import redis
import io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pytesseract

from io import BytesIO
from PIL import Image

r = redis.Redis(host='35.243.66.11', password='test')

im = Image.open('test.png')
# 
# test = int.from_bytes(im.read(), byteorder='big', signed=True)
# print(type(test))
# print(im.tobytes())
imgByteArr = io.BytesIO()
im.save(imgByteArr, format='PNG')
r.set("test", imgByteArr.getvalue())
test = Image.open(BytesIO(r.get("test")))
test.show()
# print(r.get("test"))
