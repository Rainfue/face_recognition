# Импортирование библиотек
# библиотека для API
import streamlit as st

# библиотеки для работы с картинками
import cv2

# библиотеки для моделей
from ultralytics import YOLO

#
import torch

# 
from deepface import DeepFace
#
import numpy as np
#
import pandas as pd
#
from sklearn.metrics.pairwise import cosine_similarity

#-----------------------------------------------------------------
# Инициализация дефолтных аргументов функции
# модель детекции
model = YOLO(r'C:\Users\user1\Project\face_recognition\runs\detect\face_detection_v2\weights\best.pt')

#
# device = torch.cuda.device('cuda' if torch.cuda.is_available() else 'cpu')

# тестовое фото
img = r'C:\Users\user1\Project\some_data\face_rec_data\face_ind\train\Abdullah_al-Attiyah\Abdullah_al-Attiyah_0002.jpg'

#
df = pd.read_pickle(r'C:\Users\user1\Project\face_recognition\Module_2\train.pkl')

#-----------------------------------------------------------------
# Функции
# функция рисования креста на не отработанной картинке
def draw_cross(image):
    '''
    Рисует крест на изображении.
    
    Args:
        - image: Изображение в формате numpy array.
    
    Returns:
        Изображение с нарисованным крестом.
    '''

    height, width = image.shape[:2]
    color = (255,0,0) # красный цвет креста
    thickness = 50 # толщина линий креста

    # рисуем крест
    cv2.line(image, (0,0), (width,height), color, thickness)
    cv2.line(image, (width, 0), (0, height), color, thickness)

    return image

def extract_embeddings(img_path: str, 
                       det_model: YOLO, 
                       rec_model: str = 'Facenet512'):
    '''
        Процедура для выгрузки эмбеддингов из фотографии

    Args:
        - img_path (str): Путь к изображению, на котором нужно найти цену.
        - det_model (YOLO): Модель YOLO для обнаружения bounding box'ов лиц.
        - rec_model (str): Модель распознавания для выделения эмбеддингов лиц.

    Returns:
        list: Функция возвращает ембеддинг в формате списка.
    '''   

    image = cv2.imread(img_path)
    if image is None:
        print(f"Ошибка: изображение по пути {img_path} не найдено или не может быть загружено.")
        return np.nan
    
    res = det_model.predict(image, conf=0.3, iou=0.2, device='cpu')
    # Проверка, найдены ли bounding box'ы
    if len(res[0].boxes) == 0:
        return np.nan  # Если лицо не найдено, возвращаем np.nan
    # 
    for result in res:
        boxes = result.boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1,y1,x2,y2 = map(int, box)
            #
            crop = image[y1:y2, x1:x2]

            try:
                emb_obj = DeepFace.represent(
                    img_path=crop,
                    model_name=rec_model
                )
            except Exception as e:
                print(f'Exception: {e}')
                return np.nan
            
    emb_obj = np.array(emb_obj[0]['embedding']).reshape(1,-1)
    
    return emb_obj



#-----------------------------------------------------------------
# API
# тайтл к странице приложения
st.title('Распознавание лица с фотографии')

# кнопка для скачивания справки по приложению
# with open(r'D:\Helper\MLBazyak\homework\06_01\06_01_hw\Module_A\Documentation2API.pdf', 'rb') as file:
#     st.download_button(
#         label='Справка',
#         data=file,
#         file_name='Справка.pdf',
#         mime='application/pdf'
#     )

# форма для загрузки фотографии
uploaded_file = st.file_uploader('Chose a file:', type=['jpg', 'jpeg', 'png'])

# если файл загружен, то обрабатываем его нашими функциями
if uploaded_file is not None:
    # сохраняем загруженное изображение
    with open('temp_image.jpg', 'wb') as f:
        f.write(uploaded_file.getbuffer())

    input_embed = extract_embeddings(
        img_path='temp_image.jpg',
        det_model=model
    )

    similar = 0
    name = ''
    treshold = 0.3
    out_path = ''
    for i in range(df.shape[0]):
        # Извлекаем эмбединг из списка словарей
        embedding_list = df["embedding"].iloc[i]  # Это список словарей
        if isinstance(embedding_list, list) and len(embedding_list) > 0:
            embedding = embedding_list[0]['embedding']  # Извлекаем эмбединг из первого словаря
            X = np.array(embedding).reshape(1, -1)
            
            # Вычисляем косинусное сходство
            cos_sim = float(cosine_similarity(X, input_embed))
            
            # Обновляем наиболее похожий результат
            if cos_sim > similar:
                similar = cos_sim
                name = df['name'].iloc[i]
                out_path = df['img_path'].iloc[i]
        else:
            print(f"Ошибка: неверный формат эмбединга в строке {i}")

    if similar < treshold:
        print('Неизвестная личность')
    else:
        print(f'Имя: {name}\nСходство: {similar}')

    img_rgb = cv2.cvtColor(cv2.imread('temp_image.jpg'), cv2.COLOR_BGR2RGB)
    out_rgb = cv2.cvtColor(cv2.imread(out_path), cv2.COLOR_BGR2RGB)

    # Создаём два столбца
    col1, col2 = st.columns(2)

    # Отображаем первое изображение в первом столбце
    with col1:
        st.image(
            img_rgb,
            caption='Input image',
            use_container_width=True
        )

    # Отображаем второе изображение во втором столбце
    with col2:
        st.image(
            out_rgb,
            caption=f'Out image: {name}',
            use_container_width=True
    )
