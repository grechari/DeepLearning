# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 17:27:20 2024

@author: AM4
"""
# Импортируем библиотеки
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import ultralytics
from ultralytics import YOLO
import cv2
#import matplotlib.pyplot as plt

# Проверяем что доступно из оборудования
ultralytics.checks()

# Обучаем модель
model = YOLO("yolov8s.pt")
results = model.train(data="raccoon.yaml", model="yolov8s.pt", epochs=2, batch=6,
                      project='raccoon_detection', val = True, verbose=True)


# Смотрим результат
results = model("maxresdefault.jpg")
result = results[0]
cv2.imshow("YOLOv8", result.plot())

