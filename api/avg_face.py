import json as request_json
import mimetypes
import os
import threading
import time
from logging import handlers

import numpy as np
from PIL import Image
from sanic import Sanic
from sanic.response import json
from sanic import Request
import logging
import uuid
# from configs import register, remove
import requests
import cv2
from flask import Flask, request, jsonify
from facer import facer
import base64

import matplotlib.pyplot as plt

from concurrent.futures import ThreadPoolExecutor


executor = ThreadPoolExecutor(max_workers= 20)

# 日志级别字典
__level_dict = {
    'critical': logging.CRITICAL,
    'fatal': logging.CRITICAL,
    'error': logging.ERROR,
    'warning': logging.WARNING,
    'warn': logging.WARNING,
    'info': logging.INFO,
    'debug': logging.DEBUG
}


# 实例化
# client = Minio(
# 	endpoint = "https://test-minio.toplion.com.cn",
# 	# Minio 服务器访问 key
# 	access_key= "minioadmin",
# 	# 密码
# 	secret_key= "minioadmin",
# 	# secure 指定是否以安全模式创建Minio连接
# 	# 建议为False
# 	secure= False
# )

# 封装日志方法
def get_log(filename, level, when='MIDNIGHT', backupCount=3, maxBytes=10000000,
            fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
    level = __level_dict.get(level.lower(), None)
    logger = logging.getLogger(filename)  # 设置日志名称
    format_str = logging.Formatter(fmt)  # 设置日志格式
    logger.setLevel(level)  # 设置日志级别
    console_handler = logging.StreamHandler()  # 控制台输出
    console_handler.setFormatter(format_str)  # 控制台输出的格式
    logger.addHandler(console_handler)  # 控制台输出
    file_handler = handlers.RotatingFileHandler(filename=filename, maxBytes=maxBytes, backupCount=backupCount,
                                                encoding='utf-8')  # 文件输出
    file_handler.setFormatter(format_str)  # 文件输出格式
    logger.addHandler(file_handler)  # 日志输出
    return logger


log = get_log('avg_face.log', 'info')


# def array_to_img(x, data_format=None, scale=True):
#     """Converts a 3D Numpy array to a PIL Image instance.
#
#     # Arguments
#         x: Input Numpy array.
#         data_format: Image data format.
#         scale: Whether to rescale image values
#             to be within [0, 255].
#
#     # Returns
#         A PIL Image instance.
#
#     # Raises
#         ImportError: if PIL is not available.
#         ValueError: if invalid `x` or `data_format` is passed.
#     """
#     if pil_image is None:
#         raise ImportError('Could not import PIL.Image. '
#                           'The use of `array_to_img` requires PIL.')
#     x = np.asarray(x, dtype=K.floatx())
#     if x.ndim != 3:
#         raise ValueError('Expected image array to have rank 3 (single image). '
#                          'Got array with shape:', x.shape)
#
#     if data_format is None:
#         data_format = K.image_data_format()
#     if data_format not in {'channels_first', 'channels_last'}:
#         raise ValueError('Invalid data_format:', data_format)
#
#     # Original Numpy array x has format (height, width, channel)
#     # or (channel, height, width)
#     # but target PIL image has format (width, height, channel)
#     if data_format == 'channels_first':
#         x = x.transpose(1, 2, 0)
#     if scale:
#         x = x + max(-np.min(x), 0)
#         x_max = np.max(x)
#         if x_max != 0:
#             x /= x_max
#         x *= 255
#     if x.shape[2] == 3:
#         # RGB
#         return pil_image.fromarray(x.astype('uint8'), 'RGB')
#     elif x.shape[2] == 1:
#         # grayscale
#         return pil_image.fromarray(x[:, :, 0].astype('uint8'), 'L')
#     else:
#         raise ValueError('Unsupported channel number: ', x.shape[2])


# 根据base64生成图片，并保存到对应文件夹
def base64_to_img(img_base64_list, path_to_images):
    """
    根据base64生成图片.
    :param img_base64_list: 图片的base64文件
    :param path_to_images: 要保存的文件夹相对路径
    :returns: None
    """
    # 循环转换图片到文件夹
    for index, s in enumerate(img_base64_list):
        try:
            image_info_lit = s.split(';', 2)
            ext_name = mimetypes.guess_extension(image_info_lit[0].split(':', 2)[1])
            with open(path_to_images + str(index) + ext_name, 'wb') as f:
                f.write(base64.b64decode(image_info_lit[1].split(',')[1]))
        except Exception as ex:
            print(ex)


# base64 集合转 cv2
def base64_list_to_cv2_list(img_base64_list):
    cv2_list = []
    assert img_base64_list is not None
    for index, s in enumerate(img_base64_list):
        data = base64.b64decode(s.encode('utf8'))
        data = np.frombuffer(data, np.uint8)
        data = cv2.imdecode(data, cv2.IMREAD_COLOR)
        cv2_list.append(data)
    return cv2_list


def base64_list_to_cv2_list_2(images):
    assert images != [], "images is none, Please check the input data"
    images_decode = [base64_to_cv2(image) for image in images]
    return images_decode


def cv2_list_to_img_list(cv2_list, path):
    assert cv2_list != [], "cv2_list is none, Please check the input data"
    return [mat2image(i) for i in cv2_list]


def save_image_to_path(path, imgs):
    for index, s in enumerate(imgs):
        name = path + str(index) + '.png'
        try:
            s.save(name, 'PNG')
        except Exception as ex:
            print(ex)


# 根据 url 获取网络图片并保存到磁盘
# def saveImageFromHttp(image_url_list, path_to_images,timeout_s=3):
#     for index,image_url in enumerate(image_url_list):
#         try:
#             resp = urllib.request.urlopen(image_url, timeout=timeout_s)
#             image = np.asarray(bytearray(resp.read()), dtype="uint8")
#             image = cv2.imdecode(image, cv2.IMREAD_COLOR)
#             print(resp.url)
#             cv2.imwrite(path_to_images+str(index)+'.jpg',image)
#         except Exception as error:
#             print('获取图片失败', error)


def remove_item(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):  # 如果是文件夹那么递归调用一下
            if len(os.listdir(path)) == 0:
                os.rmdir(path)
            remove_item(c_path)
        else:  # 如果是一个文件那么直接删除
            os.remove(c_path)
    if len(os.listdir(path)) == 0:
        os.rmdir(path)
    log.info('源文件已经清空完成')


def cv2_to_base64(image):
    return base64.b64encode(image).decode(
        'utf8')  # data.tostring()).decode('utf8')


def cv2_to_image(path, i):
    img = Image.open(path)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return img


def mat2image(image: np) -> Image:
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return img


def base64_to_cv2(b64str):
    data = base64.b64decode(b64str.encode('utf8'))
    data = np.frombuffer(data, np.uint8)
    data = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return data



def handle_face_task(face_images_list,tenant_type_id,unit_id,campus_id):
    print('线程名称：%s' % (threading.current_thread().name))
    # 获取当前时间戳，浮点数
    time_float = time.time()
    temp_dir = str(time_float)
    path_to_images = "../face_images/" + temp_dir + '/'
    avg_to_images = "../avg_images/" + temp_dir + '/'
    # 如果文件夹不存在则创建
    if not os.path.exists(path_to_images):
        os.makedirs(path_to_images)
    if not os.path.exists(avg_to_images):
        os.makedirs(avg_to_images)

    # base64 转为 cv2
    face_cv2_list = base64_list_to_cv2_list_2(face_images_list)
    # cv2 转为图片
    temp_images = cv2_list_to_img_list(face_cv2_list, path_to_images)
    save_image_to_path(path_to_images, temp_images)
    # 根据url获取图片并保存到文件夹
    # saveImageFromHttp(face_images_list,path_to_images)

    images = facer.load_images(path_to_images)
    landmarks, faces = facer.detect_face_landmarks(images)

    output_temp_file = str(uuid.uuid4()) + '_' + 'average_face.png'
    facer.create_average_face(faces, landmarks, save_image=True,
                              output_file=avg_to_images + output_temp_file)
    with open(avg_to_images + output_temp_file, 'rb') as f:
        img_str = f.read()
        image_base64 = str(base64.b64encode(img_str), encoding='utf-8')

    # 清空历史源数据
    remove_item(avg_to_images)
    remove_item(path_to_images)
    # 异步给Java 接口推送数据
    url = 'https://your-domamin/api/test/demo'

    if len(image_base64.strip()) > 0:
        params = {"tenantTypeId": tenant_type_id, "imageBase64": image_base64, "unitId": unit_id, "campusId": campus_id}
        headers = {'Content-Type': 'application/json'}
        data_str = request_json.dumps(params)
        requests.post(url, data=data_str, headers=headers)

# 创建一个服务，赋值给APP
# app = Sanic("AvgFaceApp")


# app.ctx.db = Database()

app = Flask(__name__)
@app.route("/avg_face/generate", methods=['POST'])
def avg_face():
    if request.json is None:
        return
    print(request.json)
    request_vo = request.json
    face_images_list = request_vo['images']
    tenant_type_id = request_vo['tenantTypeId']
    unit_id = request_vo['unitId']
    campus_id = request_vo['campusId']
    if face_images_list == [] or tenant_type_id == '' or unit_id is None or campus_id is None:
        return
    executor.submit(handle_face_task, face_images_list, tenant_type_id, unit_id, campus_id)
    return jsonify("平均脸处理中")