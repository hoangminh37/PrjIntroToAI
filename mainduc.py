import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
from tkinter import filedialog
from tkinter import *
import cv2

# Nếu cần load chuẩn hơn
model = tf.keras.models.load_model('final_model_full_data.h5')

class_labels = {
    0: 'Duong Cam',
    1: 'Cam Di Nguoc Chieu',
    2: 'Cam O To',
    3: 'Cam O To Re Phai',
    4: 'Cam O To Re Trai',
    5: 'Cam Xe May',
    6: 'Cam O To Va Xe May',
    7: 'Cam Xe Tai',
    8: 'Cam Xe Tai Tren 2.5 tan',
    9: 'Cam O To Khach Va O To Tai',
    10: 'Cam O To Ro-Mooc',
    11: 'Cam May Keo',
    12: 'Cam Xe Dap',
    13: 'Cam Xe Dap Tho',
    14: 'Cam Xe 3 va 4 Banh Tho So',
    15: 'Cam Nguoi Di Bo',
    16: 'Cam Xe Keo Day',
    17: 'Cam Xe Suc Vat Keo',
    18: 'Han Che Trong Luong Xe',
    19: 'Han Che Trong Luong Truc Xe',
    20: 'Han Che Chieu Cao Xe',
    21: 'Han Che Chieu Rong Xe',
    22: 'Han Che Chieu Dai O To',
    23: 'Han Che Chieu Dai Ro-Mooc',
    24: 'Khoang Cach Toi Thieu Giua Hai Xe',
    25: 'Dung Lai',
    26: 'Cam Re Trai',
    27: 'Cam Re phai',
    28: 'Cam Quay Dau',
    29: 'Cam O To Quay Dau',
    30: 'Cam Vuot',
    31: 'Cam O To Vuot',
    32: 'Toc Do Toi Da',
    33: 'Cam Bop Coi',
    34: 'Tram Thue Quan',
    35: 'Cam Dung Va Do Xe',
    36: 'Cam Do Xe',
    37: 'Cam Do Xe Ngay Le',
    38: 'Cam Do Xe Ngay Chan',
    39: 'Nhuong Duong Cho Xe Co Gioi Di Nguoc Chieu Trong Duong Hep',
    40: 'Het Cam Vuot',
    41: 'Het Han Che Toi Da',
    42: 'Het Tat Ca Cac Lenh Cam',
    43: 'Cam Di Thang',
    44: 'Cam Re Trai Va Phai',
    45: 'Cam Di Thang va Re Phai',
    46: 'Cam Di Thang va Re Trai',
    47: 'Cam Xe Cong Nong',
    48: 'Cho Ngoac Vong Ben Trai',
    49: 'Cho Ngoac Vong Ben Phai',
    50: 'Nhieu Cho Ngoac Nguy Hiem Lien Tiep',
    51: 'Duong Hep Hai Ben',
    52: 'Duong Hep Ben Trai',
    53: 'Duong Hep Ben Phai',
    54: 'Duong Hai Chieu',
    55: 'Duong Cat Nhau',
    56: 'Duong cat nhau trai',
    57: 'Duong cat nhau phai',
    58: 'Duong cat nhau chu T',
    59: 'Duong cat nhau chu Y',
    60: 'Duong Giao Nhau Vong Tuyen',
    61: 'Giao Nhau Voi Duong Khong Uu Tien',
    62: 'Giao Nhau Voi Duong Khong Uu Tien',
    63: 'Giao Nhau Voi Duong Khong Uu Tien',
    64: 'Giao Nhau Voi Duong Uu Tien',
    65: 'Giao Nhau Voi Tin Hieu Den',
    66: 'Giao Nhau Voi Duong Sat Co Rao Chan',
    67: 'Giao Nhau Voi Duong Sat Khong Co Rao Chan',
    68: 'Cau Hep',
    69: 'Cau Tam',
    70: 'Cau Xoay - Cau Dat',
    71: 'Ke Vuc Sau Phia Truoc',
    72: 'Duong Ngam',
    73: 'Ben Pha',
    74: 'Cua Chui',
    75: 'Doc Xuong Nguy Hiem',
    76: 'Doc Len Nguy Hiem',
    77: 'Duong Khong Bang Phang',
    78: 'Duong Tron Truot',
    79: 'Vach Nui Nguy Hiem',
    80: 'Nguoi Di bo Cat Ngang',
    81: 'Tre Em',
    82: 'Nguoi Di Xe Dap Cat Ngang',
    83: 'Cong Truong',
    84: 'Da Lo',
    85: 'Dai May bay',
    86: 'Gia Suc',
    87: 'Thu Rung Di Ngang',
    88: 'Gio Ngang',
    89: 'Nguy Hiem Khac',
    90: 'Giao Nhau Voi Duong Hai Chieu',
    91: 'Duong Doi',
    92: 'Het Duong Doi',
    93: 'Cau Vong',
    94: 'Duong Cap Dien Phia Tren',
    95: 'Duong Cao Toc Phia Truoc',
    96: 'Duong Ham',
    97: 'Cho Duong Sat Cat Duong Bo',
    98: 'Duong Sat Cat Duong Bo Khong Vuong Goc',
    99: 'Doan Duong Hay Xay Ra Tai Nan',
    100: 'Di cham',
    101: 'Vong tranh hai ben ',
    102: 'Vong tranh ben trai',
    103: 'Vong tranh ben phai ',
    104: 'Cac xe chi duoc di thang ',
    105:'Cac xe chi duoc re phai ',
    106:'Cac xe chi duoc re trai ',
    107:'Cac xe chi duoc re phai ',
    108:'Cac xe chi duoc re trai ',
    109:'Cac xe chi duoc di thang - re phai',
    110:'Cac xe chi duoc di thang - re trai',
    111:'Cac Xe Chi Duoc Re Trai va Re Phai',
    112: 'Huong Di Vong Chuong Ngai Vat Sang Phai',
    113: 'Huong Di Vong Chuong Ngai Vat Sang Trai',
    114: 'Noi Giao Nhau Chay Theo Vong Tuyen',
    115: 'Duong Danh Cho Xe Tho So',
    116: 'Duong Danh Cho Nguoi Di Bo',
    117: 'Toc Do Toi Thieu Cho Phep',
    118: 'Het Han Che Toc Do Toi Thieu',
    119: 'Tuyen Duong Cau Vuot Bat Qua',
    120: 'An Coi',
}


def upload_image():
    root = Tk()
    root.withdraw()  
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    root.destroy()
    return file_path if file_path else None

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def resize_with_padding(img, target_size=(224, 224), pad_color=(0, 0, 0)):
        h, w = img.shape[:2]
        sh, sw = target_size

        aspect = w / h
        if aspect > 1:  # ảnh ngang
            new_w = sw
            new_h = int(new_w / aspect)
        else:
            new_h = sh
            new_w = int(new_h * aspect)

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        pad_vert = (sh - new_h) // 2
        pad_horz = (sw - new_w) // 2
        padded = cv2.copyMakeBorder(resized,
                                     pad_vert, sh - new_h - pad_vert,
                                     pad_horz, sw - new_w - pad_horz,
                                     borderType=cv2.BORDER_CONSTANT,
                                     value=pad_color)
        return padded

    img_resized = resize_with_padding(img)
    img_array = img_resized.astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

img_path = upload_image()

if img_path:
    preprocessed_img = preprocess_image(img_path)
    predictions = model.predict(preprocessed_img)
    predicted_class = np.argmax(predictions, axis=1)
    print(f'Predicted Class: {predicted_class}')
    print(f'Predicted Traffic Sign: {class_labels[predicted_class[0]]}')
else:
    print("No image selected.")

