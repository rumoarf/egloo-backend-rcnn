import os
import sys
import urllib.request
import pytesseract


def translate(text, source='ja', target='ko'):
    client_id = "KBG1eI7GshNeEKHwRCoK"  # 개발자센터에서 발급받은 Client ID 값
    client_secret = "QY2cjE0kMA"  # 개발자센터에서 발급받은 Client Secret 값
    text = urllib.parse.quote(text)
    data = "source=" + source + "&target=" + target + "&text=" + text
    request = urllib.request.Request(
        "https://openapi.naver.com/v1/papago/n2mt")
    request.add_header("X-Naver-Client-Id", client_id)
    request.add_header("X-Naver-Client-Secret", client_secret)
    response = urllib.request.urlopen(request, data=data.encode("utf-8"))
    rescode = response.getcode()
    if(rescode == 200):
        return response.read().decode('utf-8')
    else:
        return "Error Code:" + rescode
