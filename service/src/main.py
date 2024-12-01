'''
Модуль, в котором описан API для микросервиса
'''
from pathlib import Path
import sys

from fastapi import FastAPI, UploadFile, Request
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

cwd = Path(__file__).parent
sys.path.append(str(cwd))
import model

rot_imgs = cwd.parent.joinpath("rot_imgs")
tmp = cwd.parent.joinpath("tmp")
rot_imgs.mkdir(exist_ok=True)
tmp.mkdir(exist_ok=True)
    

app = FastAPI()

app.mount("/images", StaticFiles(directory=rot_imgs),
          name="images")
templates = Jinja2Templates(directory=cwd.joinpath("public"))

@app.get("/")
def root_form():
    '''
    Описание:
        Загружает стартовую html-страницу, на которой описан API

    '''
    return FileResponse(cwd.joinpath("public/index.html"))


@app.get("/testing")
def upload_form():
    '''
    Описание:
        Загружает html-страницу для тестирования, на которой можно подгрузить 
        изображение, расположенное как локально, так и удалённо.

    '''
    return FileResponse(cwd.joinpath("public/upload.html"))

@app.post("/predict")
def predict(image_path: Path) -> dict:
    '''
    Основная функция API, с помощью которой можно предсказать угол поворота, а
    также сохранить повёрнутое изображение в папку rot_imgs, если не было 
    ошибок

    Аргументы
    ----------
    image_path : Path
        Путь до файла с изображением

    Возвращаемое значение
    -------
    dict: cловарь, содержащий результаты работы:
        
        - result: результат ("OK" и "Error")
        - angle: угол поворота (только если result=="OK")
        - path: название повёрнутого изображения (если result=="OK") или путь\
            до исходного изображения (result=="Error")

    '''
    result, ang, rot_img = model.predict(image_path)
    if result=="OK":
        correct_img_path = rot_imgs.joinpath(image_path.name)
        rot_img.save(correct_img_path)
        return {"result":"OK",
                "angle":ang, 
                "path":correct_img_path.name}
    return {"result":"Error", "path":str(image_path)}

@app.post("/result")
async def show_result(request: Request,
              img:UploadFile
              ):
    '''
    Функция, предсказывающая угол поворота, сохраняющая повёрнутое изображение 
    в папку rot_imgs и показывающая повёрнутое изображение

    Аргументы
    ----------
    request : Request
        Объект запроса FastAPI, содержащий информацию о текущем запросе.
    img : UploadFile
        Загруженный файл, который пользователь отправляет через форму.
    '''
    #сохраняем файл во временную папку
    tmp_path = tmp.joinpath(img.filename)
    with open(tmp_path, "wb") as f:
        read_file = await img.read()
        f.write(read_file)

    result = predict(tmp_path)
    if result.get("result")=="OK":
        return templates.TemplateResponse("result.html", {
            "request": request,
            "angle": result.get("angle"),        
            "image_url": f"images/{result.get("path")}"            
            })
    return result
