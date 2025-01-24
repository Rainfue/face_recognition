import os
import cv2
import streamlit as st
from ultralytics import YOLO
from deepface import DeepFace
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Инициализация модели и данных
model = YOLO(r'D:\Helper\MLBazyak\homework\face_recognition\runs\detect\face_detection_v2\weights\best.pt')
no_face = r'D:\Helper\MLBazyak\homework\face_recognition\Module_2\Data\no_face.jpg'
to_many = r'D:\Helper\MLBazyak\homework\face_recognition\Module_2\Data\to_many.jpg'
dont_know = r'D:\Helper\MLBazyak\homework\face_recognition\Module_2\Data\dont_know.jpg'
df = pd.read_pickle(r'D:\Helper\MLBazyak\homework\face_recognition\Module_2\Data\train.pkl')
df_path =r'D:\Helper\MLBazyak\homework\face_recognition\Module_2\Data\train.pkl'

# Функция для извлечения эмбеддингов
def extract_embeddings(img_path: str, det_model: YOLO, rec_model: str = 'Facenet512', df_marker: bool = True):
    image = cv2.imread(img_path)
    if image is None:
        print(f"Ошибка: изображение по пути {img_path} не найдено или не может быть загружено.")
        return np.nan
    
    res = det_model.predict(image, conf=0.3, iou=0.2, device='cpu')
    if len(res[0].boxes) == 0:
        return np.nan  # Если лицо не найдено
    
    if len(res[0].boxes) > 1:
        return 'to many'  # Если лиц больше одного
    
    for result in res:
        boxes = result.boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            crop = image[y1:y2, x1:x2]

            try:
                emb_obj = DeepFace.represent(img_path=crop, model_name=rec_model)
            except Exception as e:
                print(f'Exception: {e}')
                return 'no emb'
    if df_marker:       
        emb_obj = np.array(emb_obj[0]['embedding']).reshape(1, -1)
        return emb_obj
    
    else:
        if type(emb_obj) == str:
            return np.nan
        else:
            return emb_obj

# Функция для отображения изображений
def output_images(path1: str, path2: str, caption2: str):
    img_rgb = cv2.cvtColor(cv2.imread(path1), cv2.COLOR_BGR2RGB)
    out_rgb = cv2.cvtColor(cv2.imread(path2), cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)
    with col1:
        st.image(img_rgb, caption='Input image', use_container_width=True)
    with col2:
        st.image(out_rgb, caption=f'{caption2}', use_container_width=True)

def extract_all_emb(start_path: str, name: str, df: pd.DataFrame, df_path: str = df_path):
    print('---------------------------------')
    print(df.shape)
    print(int(df.shape[0]))
    print(name)
    ind = int(df.shape[0])

    for img in os.listdir(start_path):
        img_path = os.path.join(start_path, img)
        embed = extract_embeddings(img_path, model, df_marker=False)
        print(type(embed))
        # print(embed)
        if type(embed) == list:
            new_row = {
            'name': name,
            'img_path': img_path,
            'embedding': embed.tolist() if isinstance(embed, np.ndarray) else embed,
            'img_path2': img_path
                        }   
            print(new_row)
            df.loc[ind] = new_row
            ind+=1
        else:
            print('fff')
    df = df.sort_values('name', ignore_index=True).drop_duplicates(subset=['embedding'])
    df.to_pickle(f'{df_path}')
    print(df.shape)
    print('---------------------------------')

# Основное приложение Streamlit
st.title('Распознавание лица с фотографии')

# Создаем две вкладки
tab1, tab2 = st.tabs(["Распознавание", "Загрузка фотографий"])

if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []  # По умолчанию список файлов пуст

# Вкладка 1: Распознавание
with tab1:
    df = df.drop_duplicates(subset=['embedding'])
    uploaded_file = st.file_uploader('Выберите файл:', type=['jpg', 'jpeg', 'png'], key="tab1_uploader")

    if uploaded_file is not None:
        df = pd.read_pickle(r'D:\Helper\MLBazyak\homework\face_recognition\Module_2\Data\train.pkl')
        with open('temp_image.jpg', 'wb') as f:
            f.write(uploaded_file.getbuffer())

        input_embed = extract_embeddings(img_path='temp_image.jpg', det_model=model)

        if isinstance(input_embed, str) and input_embed == 'no emb':
            output_images('temp_image.jpg', no_face, 'Result')

        if isinstance(input_embed, str) and input_embed == 'to many':
            output_images('temp_image.jpg', to_many, 'Result')

        elif isinstance(input_embed, np.ndarray):
            similar = 0
            name = ''
            treshold = 0.3
            out_path = ''
            for i in range(df.shape[0]):
                embedding_list = df["embedding"].iloc[i]
                if isinstance(embedding_list, list) and len(embedding_list) > 0:
                    embedding = embedding_list[0]['embedding']
                    X = np.array(embedding).reshape(1, -1)
                    cos_sim = float(cosine_similarity(X, input_embed))
                    if cos_sim > similar:
                        similar = cos_sim
                        name = df['name'].iloc[i]
                        out_path = df['img_path2'].iloc[i]

            if similar < treshold:
                output_images('temp_image.jpg', dont_know, 'Result: В базе нет таких людей')
            else:
                output_images('temp_image.jpg', out_path, f'Result: {name}')
                st.write(f'Сходство: {similar:.4f}')


# Инициализация сессионного состояния
if 'user_name' not in st.session_state:
    st.session_state.user_name = ""  # По умолчанию поле пустое



# Вкладка 2: Загрузка фотографий
with tab2:
    st.header("Загрузка фотографий")
    
    # Поле ввода имени
    user_name = st.text_input("Введите ваше имя на английском:")

    # Форма для загрузки файлов
    uploaded_files = st.file_uploader(
        "Загрузите до 5 фотографий", 
        type=['jpg', 'jpeg', 'png'], 
        accept_multiple_files=True, 
        key="tab2_uploader"
    )

    # Если файлы загружены, обрабатываем их
    if uploaded_files:
        if user_name and uploaded_files:
            if len(uploaded_files) > 5:
                st.warning("Вы можете загрузить не более 5 фотографий.")
            else:
                # Создаем папку с именем пользователя
                user_folder = os.path.join("user_uploads", user_name)
                os.makedirs(user_folder, exist_ok=True)

                # Сохраняем фотографии локально
                for i, uploaded_file in enumerate(uploaded_files):
                    num = len(os.listdir(user_folder))
                    file_path = os.path.join(user_folder, f"{user_name}_{num + 1}.jpg")
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                
                # Извлекаем эмбеддинги и добавляем их в DataFrame
                extract_all_emb(start_path=user_folder, name=user_name, df=df)

                # Очищаем поле ввода имени и состояние загруженных файлов
                st.session_state.user_name = ""  # Сбрасываем значение в сессионном состоянии
                st.session_state.uploaded_files = []  # Очищаем список загруженных файлов

                st.success(f"Фотографии успешно загружены в папку {user_folder}.")