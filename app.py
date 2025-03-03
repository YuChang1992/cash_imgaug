import streamlit as st  # 用於快速構建和分享數據應用的 Python 庫，可以創建簡單的 Web 應用，並在其中展示數據視覺化、圖像處理、模型結果等
from skimage import data, color, io  # 用於圖像處理的 Python 庫，data 提供一些內建的示範圖像，color 用來處理顏色轉換，io 用於圖像讀取和寫入
from skimage.transform import rescale, resize, downscale_local_mean
import imageio  # imageio 用於讀寫各種格式的圖像
import numpy as np  # numpy 用於數據處理，特別是數組和矩陣操作，通常用於圖像數據處理和深度學習中的張量操作
import tensorflow as tf  # tensorflow 是一個深度學習框架，keras 是其高層 API，用於構建和訓練神經網絡模型
from tensorflow.keras.models import Sequential  # Sequential 用來構建模型的順序層
from tensorflow.keras.models import model_from_json  # model_from_json 用來從 JSON 格式載入已保存的模型結構
from imgaug import augmenters as iaa  # imgaug 是用於圖像增強的庫，
import json  # json 是一個 Python 模塊，可以將模型結構或其他設置保存為 JSON 格式，然後再加載

with open('label_dict.json', 'r') as f:  # 讀取檔案，'r' 表示以只讀模式
    label_dict = json.load(f)  # 將讀取的 JSON 格式的數據解析並轉換成 Python 對象，這裡是一個字典
print(label_dict)
    
model = tf.keras.models.load_model('cash.keras',  # 加載已經訓練好的 Keras 模型
                                   compile=False)  # 加載模型後是否進行編譯，如果只想加載模型進行預測，不需要重新編譯模型

st.title("上傳紙鈔圖片辨識")  # 標題

uploaded_file = st.file_uploader("上傳圖片(.jpg, .png)", type=["jpg","png"])  # Streamlit 內建的部件，用於讓用戶上傳文件，第一個參數為提示文字，第二個參數為指定上傳的文件類型
img_size = (300, 600, 3)  # 設定模型所需的圖片尺寸
if uploaded_file is not None:  # 如果用戶有上傳文件
    # image = imageio.v3.imread(uploaded_file)
    # resize2 = iaa.Resize({"height": 500, "width": 1024})
    # image = resize2(image=image)
    # image = image / 255.0
    # X1 = image.reshape((-1, *image.shape))
    # predictions = np.argmax(model.predict(X1))
    # st.markdown(f"# {label_dict[predictions]}")
    # st.image(image)

    image = io.imread(uploaded_file)  # 從文件中讀取圖片並將其轉換為 NumPy 陣列
    image = resize(image, img_size[:-1])  # 將圖片調整為模型所需的尺寸，只調整寬高，不調整通道數 (height, width, channels)
    # image = np.repeat(image, 3, axis=-1)  # 如果上傳的是單通道灰階圖片，將單通道圖像轉換為三通道
    X1 = image.reshape(1,*img_size) # / 255  將圖片數組重塑為模型所需的形狀(batch_size, height, width, channels)
    st.write("predict...")  # 在 Streamlit 應用中顯示一條文字消息，告訴用戶模型正在進行預測
    predictions = np.argmax(model.predict(X1))  # 將處理過的圖像 X1 輸入到模型中進行預測，np.argmax() 返回模型對該圖像的預測結果中概率最大的類別索引
    st.markdown(f"# {label_dict[str(predictions)]}")  # 使用 predictions 索引從 label_dict 字典中查找對應的標籤，st.markdown() 用來顯示標籤或文字，f"# {}" 用 Markdown 格式顯示標題級別的文本
    st.image(image)  # 在 Streamlit 應用中顯示圖片
