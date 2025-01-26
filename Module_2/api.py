# библиотека для API
import streamlit as st

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

# библиотека для работы с файловой системой 
import os

# реализованные мной функции распознавания
from function import get_photo, face_recognition

# -------------------------------------------------------------------------------

# Инициализация модели и данных
model = YOLO(r'D:\Helper\MLBazyak\homework\face_recognition\runs\detect\face_detection_v2\weights\best.pt')
# пути к различным изображениям
no_face = r'D:\Helper\MLBazyak\homework\face_recognition\Module_2\Data\no_face.jpg'
to_many = r'D:\Helper\MLBazyak\homework\face_recognition\Module_2\Data\to_many.jpg'
dont_know = r'D:\Helper\MLBazyak\homework\face_recognition\Module_2\Data\dont_know.jpg'
# инициализация датафрейма с эмбеддингами
df = pd.read_pickle(r'D:\Helper\MLBazyak\homework\face_recognition\Module_2\Data\train.pkl')
# путь к датафрейму
df_path =r'D:\Helper\MLBazyak\homework\face_recognition\Module_2\Data\train.pkl'

# -------------------------------------------------------------------------------

# Функция для отображения изображений
def output_images(path1: str, path2: str, caption2: str):
    # преобразование цветовой гаммы изображений
    img_rgb = cv2.cvtColor(cv2.imread(path1), cv2.COLOR_BGR2RGB)
    out_rgb = cv2.cvtColor(cv2.imread(path2), cv2.COLOR_BGR2RGB)
    # разделение на 2 части
    col1, col2 = st.columns(2)
    # вывод исходного изображения
    with col1:
        st.image(img_rgb, caption='Input image', use_container_width=True)
    # вывод итогового распознания
    with col2:
        st.image(out_rgb, caption=f'{caption2}', use_container_width=True)

# -------------------------------------------------------------------------------

# название страницы
st.title('Распознавание человека по фотографии')

# форма для загрузки фотографии
uploader = st.file_uploader('Выберите изображение', 
                            type=['jpg', 'jpeg', 'png', 'jfif'])

# при загрузке изображения в форму:
if uploader is not None:
    user_img = 'user_img.jpg'
    # сохраняем файл как временный
    with open(user_img, 'wb') as f:
        f.write(uploader.getbuffer())
    # пробуем найти лицо на фотографии пользователя
    crop_test = get_photo(user_img, model)
    # проверка выполнения функции
    match crop_test:
        case str() if crop_test == 'to_many':
            output_images(user_img, to_many, 'Слишком много лиц на фотографии')

        case str() if crop_test == 'no_face':
            output_images(user_img, no_face, 'Лицо на фотографии не найдено')

        case tuple():
            # получаем кроп лица
            face, path = crop_test
            # пробуем получить данные о распознавании        
            rec_test = face_recognition(face, path, df)

            # проверка выполнения функции
            match rec_test:
                # если фотография не прошла трешхолд схожести 
                # (такого лица нет в базе данных)
                case str() if rec_test == 'unknown':
                    output_images(user_img, to_many, 'Такого человека нет в базе данных')

                # если не получилось извлечь эмбеддинг
                case str() if rec_test == 'no_emb':
                    output_images(user_img, no_face, 'Не удалось извлечь эмбеддинг лица\nПопробуйте загрузить фотографию с более четким лицом')

                # если все прошло успешно
                case tuple():
                    # получаем результат распознавания
                    similar, name, path, out_path = face_recognition(face, path, df)
                    # выводим результаты
                    output_images(user_img, out_path, f'Результат распознавания: {name}')
                    
                    st.write(f'Сходство: {similar*100:.3f}')