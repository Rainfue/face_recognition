# библиотека с предобученной моделью YOLO
from ultralytics import YOLO

# библиотеки для обработки и анализа изображений
import cv2

# библиотека для модели распознавания
from deepface import DeepFace

# библиотека для работы с датафреймами
import pandas as pd

# для работы с массивами и математическими операциями
import numpy as np

# для вычисления косинусной схожести
from sklearn.metrics.pairwise import cosine_similarity

# модуль для отключения предупреждений
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------------------------
# функция для получения фотографии
def get_photo(img_path: str,
              model: YOLO) -> tuple:
    '''
        Функция для получения фотографии (для последующего распознания)

    Args:
        - img_path (str): путь к фотографии
        
    Returns:
        - tuple (np.ndarray, str)
    '''
    # проверки на правильный тип данных в модели
    if not isinstance(img_path, str):
        raise TypeError('img_path должен быть объектом str')

    if not isinstance(model, YOLO):
        raise TypeError('model должна быть объектом YOLO')

    # приводим изображение в удобный формат
    image = cv2.imread(img_path)
    # проверяем, точно ли мы получаем изображение
    if image is None:
        print(f"Ошибка: изображение по пути {img_path} не найдено или не может быть загружено.")
        return np.nan
    
    # используя модель детекции, находим лицо на изображении
    res = model.predict(image, conf=0.3, iou=0.2, verbose=False)

    # проверка, найдены ли bounding box'ы
    if len(res[0].boxes) == 0:
        print('На фотографии не было найдено лиц, попробуйте другое')
        return 'no_face'  # если лицо не найдено, возвращаем np.nan
    
    # проверка, не больше ли 1 лица найдено
    if len(res[0].boxes) > 1:
        print('Слишком много лиц, попробуйте загрузить фотографию с 1 лицом')
        return 'to_many'  # если лицо не найдено, возвращаем np.nan
    
    # проходимся по результатам детекции 
    for result in res:
        boxes = result.boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1,y1,x2,y2 = map(int, box)
            # обрезаем исходное изображение, оставляя только лицо
            crop = image[y1:y2, x1:x2]

    return (crop, img_path)

# ---------------------------------------------------------------------------------------------
# функция для отправки результатов распознавания
def face_recognition(
        crop: np.ndarray, 
        img_path: str,
        emb_df: pd.DataFrame
                     ) -> tuple:
    '''
        Функция для отправки результатов распознавания
        
    Args:
        - photo_info (np.ndarray): кроп детектированного лица
        - img_path (str): путь к изначальному изображению
        - emb_df (pd.DataFrame): датафрейм с известными эмбеддингами

    Returns:
        tuple (float, str, str, str)
    '''
    # проверки на правильный тип данных в модели
    if not isinstance(crop, np.ndarray):                                   
        raise TypeError('crop должен быть объектом np.ndarray')

    if not isinstance(img_path, str):
        raise TypeError('img_path должен быть объектом str')

    if not isinstance(emb_df, pd.DataFrame):
        raise TypeError('emb_df должен быть объектом pd.DataFrame')
    
    # вычисляем эмбеддинг входного изображения
    try:
        input_emb = DeepFace.represent(
                    img_path=crop,
                    model_name='Facenet512' # используем модель FaceNet512
                )
    # если эмбеддинг извлечь не удалось, выводим ошибку
    except Exception as e:
        print(f'Ошибка: {e}\nНевозможно извлечь эмбединг')
        return 'no_emb'

    # так как DeepFace.represent() возвращает список словарей, нужно извлечь требуемые данные
    # и привести их в двумерный массив
    input_emb = np.array(input_emb[0]['embedding']).reshape(1,-1)
    # переменная, куда будут записываться максимальное сходство
    similar = 0
    # переменная, куда будет записываться имя человека, у которого максимальное сходство
    name = ''
    # пороговое значение для сходства (сходство не может быть меньше этого значения)
    treshold  = 0.6
    # переменная, куда будет записываться путь к изображению с максимальным сходством
    out_path = ''

    # проходимся по всему датафрейму
    for i in range(emb_df.shape[0]):
        # в датафреме данные по фотографии хранятся в виде списка словарей
        emb_list = emb_df['embedding'].iloc[i] 
        # извлекаем эмбеддинг из списка словарей, и приводим его к 2d
        emb = np.array(emb_list[0]['embedding']).reshape(1,-1)
        # вычисляем косинусное сходство
        cos_sim = cosine_similarity(emb, input_emb).item()
        

        # обновляем переменные в зависимости от результата сравнения
        if cos_sim > similar:
            similar = cos_sim
            name = emb_df['name'].iloc[i]
            out_path = emb_df['img_path2'].iloc[i]

    # если изображение не прошло трешхолд, значит такого человека нет в базе данных
    if similar < treshold:
        print('Неизвестная личность') 
        return 'unknown'
    # иначе, возвращаем результаты сходства в кортеже
    else:
        return (similar, name, img_path, out_path)    
