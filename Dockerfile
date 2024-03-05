FROM ubuntu:22.04
WORKDIR /app
COPY ./api ./api
COPY ./face_lib ./face_lib
COPY ./facer ./facer
COPY ./requirements.txt /app/requirements.txt
ENV PATH="/usr/bin:${PATH}"
RUN apt-get update && apt-get install -y python3 && apt-get install -y pip && apt-get install -y libgl1-mesa-glx && apt-get install -y libglib2.0-0 libglib2.0-dev
RUN pip install --upgrade setuptools
RUN pip install --upgrade pip
RUN pip install wheel setuptools
RUN pip install pybind11 -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
RUN pip3 install cmake
# RUN pip install cmake==3.8.2 -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
# ENV PATH="/cmake-3.27.0-rc5-linux-x86_64/bin:${PATH}"
# RUN pip install opencv-python -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
RUN pip install --no-cache-dir --upgrade -r requirements.txt -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && echo 'Asia/Shanghai' >/etc/timezone
EXPOSE 8000

# sanic 启动方式
# CMD ["sanic", "api.avg_face:app", "--host", "0.0.0.0"]

# falsk 启动方式
CMD ["python3", "-m" , "flask",  "--app","/app/api/avg_face.py", "run", "--host=0.0.0.0","--port=8000"]