# Импортирование библиотек
# библиотека для API
import streamlit as st

# библиотеки для работы с картинками
import cv2

# библиотеки для моделей
from ultralytics import YOLO

#
import torch


#-----------------------------------------------------------------
# Инициализация дефолтных аргументов функции
# модель детекции
model = YOLO(r'C:\Users\user1\Project\face_recognition\runs\detect\face_detection_v2\weights\best.pt')

#
device = torch.cuda.device('cuda' if torch.cuda.is_available() else 'cpu')

# тестовое фото
img = r'C:\Users\user1\Project\some_data\face_rec_data\face_ind\train\Abdullah_al-Attiyah\Abdullah_al-Attiyah_0002.jpg'

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

def rec_price(det_model: YOLO = model, 
              img_dir: str = img):
    
    '''
        Процедура для обнаружения и распознавания цен на изображении.

    Args:
        - det_model (YOLO): Модель YOLO для обнаружения bounding box'ов цен.
        - ocr (easyocr.Reader): Модель OCR для распознавания текста.
        - img_dir (str): Путь к изображению, на котором нужно найти цену.

    Returns:
        list: Функция отображает изображение с обнаруженными ценами и возвращает список с определенными ценами
    '''    
    # проверки на правильный тип данных в модели
    if not isinstance(det_model,YOLO):                                   
        raise TypeError('det_model должена быть объектом YOLO')

    if not isinstance(img_dir,str):
        raise TypeError('img_dir должен быть путем к фотографии типа str')

    # загружаем фотографии для модели детекции
    image = cv2.imread(img_dir)
    res = det_model.predict(image, conf=0.3, iou=0.1, device=device)

    # проходимся по результатами модели
    for result in res:
        boxes = result.boxes.xyxy.cpu().numpy()
        for box in boxes:
            # находим x и y боксов, которые определила модель детекции
            x1, y1, x2, y2 = map(int, box)

            cv2.rectangle(image, (x1,y1), (x2,y2), (97, 255, 0), 2)   
            cv2.putText(image, 'face', (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 97, 0), 2)

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # возвращаем rgb изображение 
    return img_rgb

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

    img_rgb = rec_price(img_dir='temp_image.jpg')

    # выводим изображение с найденными (или не найденными) ценниками
    st.image(img_rgb,
             caption='Обнаруженные лица',
             use_container_width=True
             )
