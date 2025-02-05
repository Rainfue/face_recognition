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
def get_photo(image: np.ndarray, model: YOLO) -> tuple:
    '''
    Функция для получения фотографии (для последующего распознания)

    Args:
        - image (np.ndarray): изображение в формате NumPy массива
        - model (YOLO): модель YOLO для детекции лиц

    Returns:
        - tuple: (crop, img_boxes), где:
            - crop (np.ndarray): обрезанное изображение лица
            - img_boxes (np.ndarray): исходное изображение с выделенным bounding box
    '''
    # Проверки на правильный тип данных
    if not isinstance(image, np.ndarray):
        raise TypeError('image должен быть объектом np.ndarray')

    if not isinstance(model, YOLO):
        raise TypeError('model должна быть объектом YOLO')

    # Используем модель для детекции лиц
    res = model.predict(image, conf=0.3, iou=0.2, verbose=False)

    # Проверка, найдены ли bounding box'ы
    if len(res[0].boxes) == 0:
        print('На фотографии не было найдено лиц, попробуйте другое')
        return 'no_face'  # Если лицо не найдено, возвращаем строку

    # Проверка, не больше ли 1 лица найдено
    if len(res[0].boxes) > 1:
        print('Слишком много лиц, попробуйте загрузить фотографию с 1 лицом')
        return 'to_many'  # Если лицо не найдено, возвращаем строку

    # Создаем копию изображения для отрисовки bounding box
    img_boxes = image.copy()

    # Проходимся по результатам детекции
    for result in res:
        boxes = result.boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            # Обрезаем исходное изображение, оставляя только лицо
            crop = image[y1:y2, x1:x2]
            # Отрисовываем bounding box на изображении
            cv2.rectangle(img_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return (crop, img_boxes)

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
