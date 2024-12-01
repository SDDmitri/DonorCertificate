'''
Модуль для предсказания угла поворта изображения
'''
from pathlib import Path

import torch
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.functional import (to_pil_image,
                                               InterpolationMode,
                                               rotate)

DEVICE_TYPE = "cpu"

def load_model(param_path:Path):
    ''' 
    Функция для загрузки весов для модели resnet34
    
    Аргументы:
    ----------
    param_path : Path
        Путь до файла с весами модели resnet34

    Возвращаемое значение
    -------
    torchvision.models.resnet.ResNet:
        Модель с подгруженными весами

    '''
    model = resnet34(weights=None)
    model.fc = torch.nn.Linear(512, 4)
    model = model.to(DEVICE_TYPE)
    model.load_state_dict(torch.load(param_path,
                                     map_location=torch.device(DEVICE_TYPE),
                                     weights_only=True))
    return model

def predict(path:Path):
    '''
    Функция для предсказания угла поворота для изображения и поворот его на 
    необходимый угол для корректного распознавания

    Аргументы:
    ----------
    path : Path
        Путь до файла изображения

    Возвращаемое значение: 
    -------
    Кортеж, содержащий:
        - статус работы функции: 'Fail' - в процессе работы возникли ошибки, 
            'ОК' - корректная работы (str)
        - угол поворота в градусах (int), None - если возникли ошибки
        - повёрнутое изображение (PIL.Image), None - если возникли ошибки

    '''
    try:
        orig_img = read_image(path, mode=ImageReadMode.RGB).to(DEVICE_TYPE)
    except:
        return 'Fail', None, None
    transformer = ResNet34_Weights.IMAGENET1K_V1.transforms()
    image = transformer(orig_img)
    model = load_model(path.parents[1].joinpath('src','resnet_state_dict.pt'))
    with torch.no_grad():
        model.eval()
        preds = model(image.unsqueeze(0))
    ang = 90*preds.argmax(dim=1).item()
    rotate_img = to_pil_image(rotate(orig_img,
                                     -ang,
                                     interpolation=InterpolationMode.BILINEAR,
                                     expand=True
                                     ))

    return 'OK', ang, rotate_img
