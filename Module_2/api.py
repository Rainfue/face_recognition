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

import tempfile

# -------------------------------------------------------------------------------

# Инициализация модели и данных
model = YOLO(r'face_recognition\runs\detect\face_detection_v2\weights\best.pt')
# пути к различным изображениям
no_face = r'face_recognition\Module_2\Data\no_face.jpg'
to_many = r'face_recognition\Module_2\Data\to_many.jpg'
dont_know = r'face_recognition\Module_2\Data\dont_know.jpg'
# инициализация датафрейма с эмбеддингами
df = pd.read_pickle(r'face_recognition\Module_2\Data\train.pkl')
# путь к датафрейму
df_path =r'face_recognition\Module_2\Data\train.pkl'

# -------------------------------------------------------------------------------

# Функция для отображения изображений
def output_images(user_img, path2: str, caption2: str):
    # преобразование цветовой гаммы изображений
    # img_rgb = cv2.cvtColor(user_img, cv2.COLOR_BGR2RGB)
    out_rgb = cv2.cvtColor(cv2.imread(path2), cv2.COLOR_BGR2RGB)
    # разделение на 2 части
    col1, col2 = st.columns(2)
    # вывод исходного изображения
    with col1:
        st.image(user_img, caption='Input image', use_container_width=True)
    # вывод итогового распознания
    with col2:
        st.image(out_rgb, caption=f'{caption2}', use_container_width=True)

# -------------------------------------------------------------------------------

# название страницы
st.title('Распознавание человека по фотографии')

# кнопка для скачивания справки по приложению
with open(r'face_recognition\Module_2\Documentation.pdf', 'rb') as file:
    st.download_button(
        label='Справка',
        data=file,
        file_name='Справка.pdf',
        mime='application/pdf'
    )

# форма для загрузки фотографии
uploader = st.file_uploader('Выберите изображение или видео', 
                            type=['jpg', 'jpeg', 'png', 'jfif', "mp4", "avi"])

# при загрузке изображения в форму:
if uploader is not None:

    # Получаем имя файла
    file_name = uploader.name
    print(file_name)

    # Проверяем расширение файла
    file_extension = os.path.splitext(file_name)[1].lower()  # Извлекаем расширение и приводим к нижнему регистру
    print(file_extension)
    # Проверяем, является ли файл изображением
    if file_extension in ['.jpg', '.jpeg', '.png', '.jfif']:
            print(file_extension)
            user_img = 'user_img.jpg'
            # сохраняем файл как временный
            with open(user_img, 'wb') as f:
                f.write(uploader.getbuffer())
            user_image = cv2.imread(user_img)

            if user_image is None:
                st.error("Ошибка: изображение не может быть загружено.")
            else:
                # пробуем найти лицо на фотографии пользователя
                crop_test = get_photo(user_image, model)
                # проверка выполнения функции
                match crop_test:
                    case str() if crop_test == 'to_many':
                        output_images(user_img, to_many, 'Слишком много лиц на фотографии')

                    case str() if crop_test == 'no_face':
                        output_images(user_img, no_face, 'Лицо на фотографии не найдено')

                    case tuple():
                        # получаем кроп лица
                        face, img_boxes = crop_test
                        img_boxes = cv2.cvtColor(img_boxes, cv2.COLOR_BGR2RGB)
                        # пробуем получить данные о распознавании        
                        rec_test = face_recognition(face, 'test', df)

                        # проверка выполнения функции
                        match rec_test:
                            # если фотография не прошла трешхолд схожести 
                            # (такого лица нет в базе данных)
                            case str() if rec_test == 'unknown':
                                output_images(img_boxes, dont_know, 'Такого человека нет в базе данных')

                            # если не получилось извлечь эмбеддинг
                            case str() if rec_test == 'no_emb':
                                output_images(img_boxes, no_face, 'Не удалось извлечь эмбеддинг лица\nПопробуйте загрузить фотографию с более четким лицом')

                            # если все прошло успешно
                            case tuple():
                                # получаем результат распознавания
                                similar, name, _, out_path = rec_test
                                # выводим результаты
                                output_images(img_boxes, dont_know, f'Результат распознавания: {name}')
                                
                                st.write(f'Сходство: {similar*100:.3f}')

    elif file_extension in [".mp4", ".avi"]:

        name_place = st.empty()

        print(file_extension)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(uploader.read())
            video_path = temp_file.name

            # Открываем видео
            video = cv2.VideoCapture(video_path)
            if not video.isOpened():
                st.error("Ошибка: видео не может быть открыто.")
                st.stop()

            # Создаем VideoWriter для сохранения обработанного видео
            output_path = "output_video.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(video.get(cv2.CAP_PROP_FPS))
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            # Переменная для ограничения количества вызовов face_recognition
            tries = 0
            counter = 0
            # Место для отображения видео
            video_placeholder = st.empty()

            # Обработка видео
            while video.isOpened():
                ret, frame = video.read()
                if not ret:
                    break

                # Обрабатываем кадр
                result = get_photo(frame, model)

                if isinstance(result, tuple):
                    crop, img_boxes = result

                    if counter == 10:
                        tries = 0
                        counter = 0 

                    # Распознаем лицо (если tries == 0)
                    if tries == 0:
                        test_rec = face_recognition(crop, 'test_path', df)
                        if isinstance(test_rec, tuple):
                            similar, name, _, _ = test_rec
                            tries += 1
                            name_place.write(f'Распознанное имя: {name}\nСхожесть: {similar}')

                            # Добавляем текст на кадр
                    cv2.putText(img_boxes, f"{name} ({similar:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Сохраняем кадр в выходное видео
                    out.write(img_boxes)

                    # Отображаем кадр в Streamlit
                    video_placeholder.image(img_boxes, channels="BGR", use_container_width=True)

                    counter+=1

            # Освобождаем ресурсы
            video.release()
            out.release()
            





# =----------=--=-=-===============-------------------------------------------------------
