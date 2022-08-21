# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 15:00:07 2020

@author: 12437
"""

import numpy as np 
import cv2
import os

def divide_method(img,m,n):#分割成m行n列
    h, w = img.shape[0],img.shape[1] # 读取图像shape
    gx, gy = np.meshgrid(np.linspace(0, w, n),np.linspace(0, h, m)) # 对图像进行切片操作
    gx=np.round(gx).astype(np.int) # 将切片后数据格式转化整数
    gy=np.round(gy).astype(np.int)

    divide_image = np.zeros([m-1, n-1, int(h*1.0/(m-1)+0.5), int(w*1.0/(n-1)+0.5),3], np.uint8)#这是一个五维的张量，前面两维表示分块后图像的位置（第m行，第n列），后面三维表示每个分块后的图像信息
#    循环给每个分块赋值
    for i in range(m-1):
        for j in range(n-1):      
            divide_image[i,j,0:gy[i+1][j+1]-gy[i][j], 0:gx[i+1][j+1]-gx[i][j],:]= img[
                gy[i][j]:gy[i+1][j+1], gx[i][j]:gx[i+1][j+1],:]#这样写比a[i,j,...]=要麻烦，但是可以避免网格分块的时候，有些图像块的比其他图像块大一点或者小一点的情况引起程序出错
    return divide_image

def save_subimg(divide_image,resultpath):#
    m,n=divide_image.shape[0],divide_image.shape[1]
    imgname = resultpath.split("\\")[-1] # 原图图名
    postfix = ".png" # 图的后缀名称
    for i in range(m):
        for j in range(n):
            subimg = divide_image[i,j,:] # 将分割后的子图逐步输出
            subimg_name = resultpath + "\\" + imgname + str(i) + str(j) + postfix # + postfix # 组装得到新的子图名称，ij即为分块位置
            cv2.imwrite(subimg_name,subimg)
            
    return None

## 运行函数
def main_div_img(num_cases,img_path,p2p_or_p2pHD):
    if p2p_or_p2pHD == "p2p":
        div_H,div_W = 5,9 #高度和宽度方向划分的次数，划分后高度分为(div_H-1),宽度划分为(div_W-1)
        img_size = (1024,512)
    elif p2p_or_p2pHD == "p2pHD":
        div_H,div_W = 5,9 #高度和宽度方向划分的次数，划分后高度分为(div_H-1),宽度划分为(div_W-1)
        img_size = (1024,512)
    elif p2p_or_p2pHD == "ManiGAN":
        div_H,div_W = 5,5 #高度和宽度方向划分的次数，划分后高度分为(div_H-1),宽度划分为(div_W-1)
        img_size = (256,256)
    else:
        print ("Error: no p2p_or_p2pHD !")
        
    for img_NO in range(num_cases):
        if p2p_or_p2pHD == "p2p":
            imgname_output = os.path.join(img_path,"test (%d)-outputs.png" %(img_NO+1))
            imgname_target = os.path.join(img_path,"test (%d)-targets.png" %(img_NO+1))
        elif p2p_or_p2pHD == "p2pHD":
            imgname_output = os.path.join(img_path,"test (%d)_synthesized_image.png" %(img_NO+1))
            imgname_target = os.path.join(img_path,"test (%d).png" %(img_NO+1))
        elif p2p_or_p2pHD == "ManiGAN":
            imgname_output = os.path.join(img_path,"test (%d)_synthesized_image.png" %(img_NO+1))
            imgname_target = os.path.join(img_path,"test (%d).png" %(img_NO+1))
        else:
            print ("Error: no p2p_or_p2pHD !")
            imgname_output = " "
            imgname_target = " "
        
        subimg_path_out = os.path.join(img_path,"test (%d)" %(img_NO+1)+"-outputs")
        subimg_path_tar = os.path.join(img_path,"test (%d)" %(img_NO+1)+"-targets")
        
        if os.path.exists(imgname_output):
            if not os.path.exists(subimg_path_out):
                os.makedirs(subimg_path_out)
            img = cv2.imread(imgname_output)
            img_resize = cv2.resize (img,img_size)
#            cv2.imwrite("shearwall (%d)-targets-RE.png"%(img_NO+1),img_resize)
            #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            divide_images = divide_method(img_resize,div_H,div_W) #图像分块
            save_subimg (divide_images,subimg_path_out) #图像保存
        if os.path.exists(imgname_target):
            if not os.path.exists(subimg_path_tar):
                os.makedirs(subimg_path_tar)
            img = cv2.imread(imgname_target)
            img_resize = cv2.resize (img,img_size)
#            cv2.imwrite("shearwall (%d)-targets-RE.png"%(img_NO+1),img_resize)
            #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            divide_images = divide_method(img_resize,div_H,div_W) #图像分块
            save_subimg (divide_images,subimg_path_tar) #图像保存
            
    return None