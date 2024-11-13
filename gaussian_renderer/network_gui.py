import json
import socket
import traceback

import torch

from scene.cameras import MiniCam

host = "127.0.0.1"
port = 6009

conn = None
addr = None

listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

def init(wish_host, wish_port):
    global host, port, listener # 意味着函数内可以对全局变量进行修改
    host = wish_host
    port = wish_port
    listener.bind((host,port))
    listener.listen()
    listener.settimeout(0)

def try_connect():
    global conn, addr, listener
    try:
        conn, addr = listener.accept()
        print(f"\nConnected by {addr}")
        conn.settimeout(None)
    # 错误会被忽略
    except Exception as inst:
        pass

def read():
    global conn
    messageLength = conn.recv(4) # 接收4个字节的信息
    messageLength = int.from_bytes(messageLength, 'little') # 小端法转换为整数
    message = conn.recv(messageLength)
    return json.loads(message.decode("utf-8")) # 解析json字符串

def send(message_bytes, verify):
    global conn
    if message_bytes != None:
        conn.sendall(message_bytes)
    conn.sendall(len(verify).to_bytes(4,'little'))
    conn.sendall(bytes(verify,'ascii'))

def receive():
    message = read()

    width = message["resolution_x"]
    height = message["resolution_y"]

    if width != 0 and height != 0:
        try:
            do_training = bool(message["train"])
            fovy = message["fov_y"]
            fovx = message["fov_x"]
            znear = message["z_near"]
            zfar = message["z_far"]
            do_shs_python = bool(message["shs_python"])
            do_rot_scale_python = bool(message["rot_scale_python"])
            keep_alive = bool(message["keep_alive"])
            scaling_modifier = message["scaling_modifier"]
            # 将 view_matrix 转换为 4x4 的张量并发送到 GPU
            world_view_transform = torch.reshape(torch.tensor(message["view_matrix"]),(4,4)).cuda()
            world_view_transform[:,1] = -world_view_transform[:,1] # ?
            world_view_transform[:,2] = -world_view_transform[:,2] # ?
            full_proj_transform = torch.reshape(torch.tensor(message["view_projection_matrix"]),(4,4)).cuda()
            full_proj_transform[:,1] = -full_proj_transform[:,1] # ?
            custom_cam = MiniCam(width,height,fovy,fovx,znear,zfar,world_view_transform,full_proj_transform)
        except Exception as e:
            print("")
            traceback.print_exc() # 用来打印异常的详细追踪信息
            raise e
        return custom_cam, do_training, do_shs_python, do_rot_scale_python, keep_alive, scaling_modifier
    else:
        return None, None, None, None, None, None

