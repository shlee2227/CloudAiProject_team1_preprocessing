import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import pandas as pd
import requests
from PIL import Image
from io import BytesIO

import os
import base64

# -------------------------세팅 영역-------------------------
# WebDriver 초기화
options = webdriver.ChromeOptions()
# options.add_argument("--headless")
options.add_argument("--window-size=1920,1080")

# driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
driver.implicitly_wait(3)


# -------------------------함수 영역-------------------------
def scroll_to_bottom():
  """페이지 끝까지 스크롤."""
  last_height = driver.execute_script("return document.body.scrollHeight")
  while True:
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)  # 페이지가 로드되는 시간을 줌
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
      break
    last_height = new_height


def get_img(data):
  img_container = driver.find_element(By.CLASS_NAME, "wIjY0d")
  img_li = img_container.find_elements(By.TAG_NAME, "h3")
  print(len(img_li))
  for li in img_li:
    img_element = li.find_element(By.TAG_NAME, "img")
    img = img_element.get_attribute("src")
    data.append({"img": img}) # 확장성을 위해 dict 자료형 사용


def img_save(idx, item):
  img_url = item["img"]
  if img_url.startswith('data:image'):
      # Base64 데이터 URI 처리
      header, encoded = img_url.split(',', 1)
      data = base64.b64decode(encoded)
      img = Image.open(BytesIO(data))
  else:
      # 일반 URL 처리
      res = requests.get(img_url)
      if res.status_code == 200:
          img = Image.open(BytesIO(res.content))
      else:
          print(f"Failed to download image at {img_url}")
          return
  if img.mode != 'RGB':
        img = img.convert('RGB')
  img = resize_and_crop_center(img, (224, 224)) # resize
  img.save(f'images/image_{idx}.jpg')


def resize_and_crop_center(img, size=(224, 224)):
  try:
    # 원본 이미지 크기
    img_width, img_height = img.size
    # 리사이즈할 비율 계산
    target_width, target_height = size
    aspect_ratio = min(img_width / target_width, img_height / target_height)
    # 리사이즈할 크기 계산
    new_width = int(img_width / aspect_ratio)
    new_height = int(img_height / aspect_ratio)
    # 이미지 리사이즈
    img = img.resize((new_width, new_height), Image.LANCZOS)
    # 크롭할 중심 좌표 계산
    left = (new_width - target_width) // 2
    top = (new_height - target_height) // 2
    right = left + target_width
    bottom = top + target_height
    # 이미지 크롭
    img = img.crop((left, top, right, bottom))
  except Exception as e:
    print(f"Error resizing image: {e}")
    raise
  return img

# -------------------------실행 영역-------------------------
url = "https://www.google.com/search?as_st=y&as_q=maltese+dog+photo&as_epq=&as_oq=&as_eq=&imgsz=m&imgar=&imgcolor=&imgtype=&cr=&as_sitesearch=&as_filetype=&tbs=&udm=2"
driver.get(url)
time.sleep(2)
scroll_to_bottom()

data = []
get_img(data)
for idx, img in enumerate(data):
  img_save(idx, img)

# WebDriver 종료
driver.quit()