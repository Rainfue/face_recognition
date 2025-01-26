# библиотека, с реализоваными unit тестами
import unittest

# импортируем функцию из нашего модуля с API
from function import get_photo, face_recognition

# библиотека для математических операций
import numpy as np

# библиотека с предобученной моделью YOLO
from ultralytics import YOLO

# библиотека для работы с датафреймами
import pandas as pd

# модуль для отключения предупреждений
import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------------------
# инициализация модели детекции
model = YOLO(r'D:\Helper\MLBazyak\homework\face_recognition\runs\detect\face_detection_v2\weights\best.pt')

# фотография с лицом
face = r'D:\Helper\MLBazyak\homework\face_recognition\Module_2\Data\putin.jpg'
# фотография без лица
no_face = r'D:\Helper\MLBazyak\homework\face_recognition\Module_2\Data\no_face_test.jfif'
# фотография с более чем 1 лицом
to_many_face = r'D:\Helper\MLBazyak\homework\face_recognition\Module_2\Data\to_many_test.jfif'
# фотография с неизвестным лицом
unknown = r'D:\Helper\MLBazyak\homework\face_recognition\Module_2\Data\uknown.jpg'
# фотография из которой не извлекается эмбеддинг
no_emb = r'D:\Helper\MLBazyak\homework\face_recognition\Module_2\Data\mikli.png'

# инициализируем датафрейм с эмбеддингами
df = pd.read_pickle(r'D:\Helper\MLBazyak\homework\face_recognition\Module_2\Data\train.pkl')


# функция get_photo()
class TestGetPhoto(unittest.TestCase):
    # тестирование случая, когда на фотографии есть лицо
    def test_face_true(self):
        self.assertTrue(type(get_photo(img_path=face, model=model)) == tuple)
        self.assertTrue(type(get_photo(img_path=face, model=model)[0]) == np.ndarray)
        self.assertTrue(type(get_photo(img_path=face, model=model)[1]) == str)

    # тестирование случая, когда на фотографии нет лица
    def test_face_false(self):
        self.assertTrue(type(get_photo(img_path=no_face, model=model)) == str)
        self.assertTrue(get_photo(img_path=no_face, model=model) == 'no_face')

    # тестирование случая, когда на фотографии больше 1 лица
    def test_to_many_face(self):
        self.assertTrue(type(get_photo(img_path=to_many_face, model=model)) == str)
        self.assertTrue(get_photo(img_path=to_many_face, model=model) == 'to_many')

    # проверка типов данных, принимаемых функцией get_photo()
    def test_data_type(self):
        # неправильно указан путь к фото
        with self.assertRaises(TypeError):
            get_photo(img_path=123, model=model)
        # неправильно указана модель детекции
        with self.assertRaises(TypeError):
            get_photo(img_path=face, model=123)


# функция face_recognition()
class TestFaceRecognition(unittest.TestCase):
    # тестирование случая, когда эмбеддинг успешно найден
    def test_emb_true(self):
        img_info = get_photo(face, model)
        self.assertTrue(type(face_recognition(img_info[0], img_info[1], df)) == tuple)
        self.assertTrue(type(face_recognition(img_info[0], img_info[1], df)[0]) == float)
        self.assertTrue(type(face_recognition(img_info[0], img_info[1], df)[1]) == str)
        self.assertTrue(type(face_recognition(img_info[0], img_info[1], df)[2]) == str)
        self.assertTrue(type(face_recognition(img_info[0], img_info[1], df)[3]) == str)

    # тестирование случая, при котором эмбеддинг находится, но в базе данных нет человека
    # со сохдстовом больше порогового (т.е. неизвестный нам человек)
    def test_emb_unknown(self):
        img_info = get_photo(unknown, model)
        self.assertTrue(type(face_recognition(img_info[0], img_info[1], df)) == str)
        self.assertTrue(face_recognition(img_info[0], img_info[1], df) == 'unknown')
    
    # тестирование случая, при котором не удается извлечь эмбеддинг
    def test_no_emb(self):
        img_info = get_photo(no_emb, model)
        self.assertTrue(type(face_recognition(img_info[0], img_info[1], df)) == str)
        self.assertTrue(face_recognition(img_info[0], img_info[1], df) == 'no_emb')

    # проверка на правильность типов данных, передоваемых функции
    def test_data_type(self):
        img_info = get_photo(face, model)
        # неправильно указан путь к фото
        with self.assertRaises(TypeError):
            get_photo(crop=img_info[0], img_path=123, emb_df=df)

        # неправильно указан путь к фото
        with self.assertRaises(TypeError):
            get_photo(crop=123, img_path=img_info[1], emb_df=df)

        # неправильно указан путь к фото
        with self.assertRaises(TypeError):
            get_photo(crop=img_info[0], img_path=img_info[1], emb_df=123)
        


if __name__ == "__main__":
    unittest.main()