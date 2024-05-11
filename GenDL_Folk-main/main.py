from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

# gpu 번호 사용법
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
'''

'''
# GPU 사용을 원하는 경우
with tf.device('/device:GPU:0'): 
    # 원하는 코드 작성(들여쓰기 필수)

# CPU 사용을 원하는 경우
with tf.device('/cpu:0'): 
    # 원하는 코드 작성(들여쓰기 필수)
'''