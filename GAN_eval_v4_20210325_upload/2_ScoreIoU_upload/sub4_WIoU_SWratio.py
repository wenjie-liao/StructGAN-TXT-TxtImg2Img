# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 21:05:53 2020

@author: Administrator
"""

import os
import cv2
import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from sklearn.metrics import confusion_matrix    # 生成混淆矩阵函数
from tensorflow.python.ops import math_ops, array_ops

## 定义所需函数
### 根据像素点颜色判断所属类别
def switch_pixel(h, s, v):
    if ((h>=0 and h<10) or (h>156 and h<=180)) and (s>=43 and s<=255) and (v>=46 and v<=255): # 红色
        return 1,h,s,v # 剪力墙类别
    elif (h>=0 and h<=180) and (s>=0 and s<43) and (v>=46 and v<=220): # 灰色
        return 2,h,s,v #普通墙类别
    elif (h>=35 and h<=77) and (s>=43 and s<=255) and (v>=46 and v<=255): # 绿色
        return 3,h,s,v #门窗类别
    elif (h>=100 and h<=124) and (s>=43 and s<=255) and (v>=46 and v<=255): # 蓝色
        return 4,h,s,v #户外门洞类别
    else:
        return 0,h,s,v #背景类别
    
### 判断图像中所有像素点的类别
def switch_image(array): 
    array_new = np.zeros((array.shape[0],array.shape[1]))
    array_newS = np.zeros((array.shape[0],array.shape[1],3)) #新的剪力墙的矩阵
    array_newI = np.zeros((array.shape[0],array.shape[1],3)) #新的填充墙的矩阵
    pixel_numS,pixel_numI = 0,0 #统计像素点个数
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            array_newS[i][j][0],array_newS[i][j][1],array_newS[i][j][2] = 0,0,255 #白色
            array_newI[i][j][0],array_newI[i][j][1],array_newI[i][j][2] = 0,0,255 #白色
            array_new[i][j],h,s,v = switch_pixel(array[i][j][0], array[i][j][1], array[i][j][2])
            if array_new[i][j] == 1: #剪力墙
                array_newS[i][j][0],array_newS[i][j][1],array_newS[i][j][2] = h,s,v
                pixel_numS += 1
            elif array_new[i][j] == 2: #普通墙
                array_newI[i][j][0],array_newI[i][j][1],array_newI[i][j][2] = h,s,v
                pixel_numI += 1
                
    return array_new,array_newS,array_newI,pixel_numS,pixel_numI

### WIoU, MIoU, PA evaluation based on confusion matrix
def eval_perform(cm_array, name):
    """Compute the mean intersection-over-union via the confusion matrix."""
    # Transfer numpy to tensor
    cm_tensor = tf.convert_to_tensor(cm_array)
#    print("cm_tensor=",type(cm_tensor))
    
    # Compute using tensor
    sum_over_row = math_ops.to_float(math_ops.reduce_sum(cm_tensor, 0))
    sum_over_col = math_ops.to_float(math_ops.reduce_sum(cm_tensor, 1))
    cm_diag = math_ops.to_float(array_ops.diag_part(cm_tensor)) # 交集,对角线值
    denominator = sum_over_row + sum_over_col - cm_diag # 分母，即并集
    
    # The mean is only computed over classes that appear in the label or prediction tensor. If the denominator is 0, we need to ignore the class.
    num_valid_entries = math_ops.reduce_sum(math_ops.cast(math_ops.not_equal(denominator, 0), dtype=tf.float32)) # 类别个数
    
    ## for IoU
    # If the value of the denominator is 0, set it to 1 to avoid zero division.
    denominator = array_ops.where(math_ops.greater(denominator, 0), denominator, array_ops.ones_like(denominator))
    iou = math_ops.div(cm_diag, denominator) # 各类IoU
    
    # If the number of valid entries is 0 (no classes) we return 0.
    miou_tensor = array_ops.where(math_ops.greater(num_valid_entries, 0),math_ops.reduce_sum(iou, name=name) / num_valid_entries, 0) #mIoU
    
    # weight iou by liaowj
    weight1 = 0.4
    weight2 = 0.4
    weight3 = 0.1
    weight4 = 0.1
    weight0 = 0.0
    Wiou_tensor = weight0*iou[0] + weight1*iou[1] + weight2*iou[2] + weight3*iou[3] + weight4*iou[4]
    
    ## for PA: pixel accuracy
    PA_tensor = math_ops.div(math_ops.reduce_sum(cm_diag), math_ops.reduce_sum(sum_over_row))
    
#    创建session，执行计算
    sess = tf.Session()
    sess.run(miou_tensor)
    # tensor转化为numpy数组
#    sum_over_row_array = sum_over_row.eval(session=sess)
#    cm_diag_array = cm_diag.eval(session=sess)
    ious_array = iou.eval(session=sess)
    miou_array = miou_tensor.eval(session=sess)
    Wiou_array = Wiou_tensor.eval(session=sess)
    PA_array = PA_tensor.eval(session=sess)
    
    return ious_array, miou_array, Wiou_array, PA_array

### 得到WIoU, MIoU, PA
def get_WIoU(arrayR2,arrayP2,imgname_target,imgname_output,img_size,result_dir,img_NO,MIoUs,WIoUs,PAs):
    # 将HSV矩阵reshape
    arrayR3 = np.reshape(arrayR2,(-1))
    arrayP3 = np.reshape(arrayP2,(-1))
    
    ## IoU计算部分
    num_cla = 5 # 像素的类别一共5类
    
    # confusion matrix
    conf_matrix=confusion_matrix(arrayR3, arrayP3)
    
    # computing IoU
    #mean_iou = tf.metrics.mean_iou(arrayR3, arrayP3, num_cla)
    name = 'useless'
    IoUs_confusematrix, MIoU, WIoU, PA = eval_perform(conf_matrix, name)
    
    pixIoU_txtname = result_dir + "\\pixIoU(%d).txt"%(img_NO+1)
    with open(pixIoU_txtname, "w+") as pixIoU_txt:
        pixIoU_txt.write("MIoU:" + str(MIoU) +"\n")
        MIoUs.append(MIoU)
        pixIoU_txt.write("Wiou:" + str(WIoU) +"\n")
        WIoUs.append(WIoU)
        pixIoU_txt.write("Pixel acc:" + str(PA) +"\n")
        PAs.append(PA)
        pixIoU_txt.write("IoUs_confusematrix:" + "\n")
        for iou in IoUs_confusematrix:
            pixIoU_txt.write(str(iou) + "\n")
            
    return MIoUs,WIoUs,PAs

### SWratio evaluation
def StoIratio(result_dir_SW,img_hsv2_S,img_hsv2_I,areaS,areaI,out_or_tar):
    SWratio = areaS/(areaI+areaS) #剪力墙在总墙体中的面积占比
    img_hsv3_S,img_hsv3_I = np.array(img_hsv2_S,dtype='uint8'),np.array(img_hsv2_I,dtype='uint8')
    img_bgr2_S = cv2.cvtColor(img_hsv3_S, cv2.COLOR_HSV2BGR)
    img_bgr2_I = cv2.cvtColor(img_hsv3_I, cv2.COLOR_HSV2BGR)
    if not os.path.exists(result_dir_SW):
        os.mkdir(result_dir_SW)
    cv2.imwrite(result_dir_SW+"\\"+out_or_tar+"_shearwall.png",img_bgr2_S) # 输出剪力墙图
    cv2.imwrite(result_dir_SW+"\\"+out_or_tar+"_infillwall.png",img_bgr2_I) # 输出剪力墙图
    
    return SWratio

### output SWratio
def SWoutput_txt(ratios,txtpath):
    mean_ratios = np.mean(ratios)
    std_ratios = np.std(ratios)  
    txtmeanstd=open(txtpath,"w+")
    txtmeanstd.write("mean value: %f"%mean_ratios + "\n")
    txtmeanstd.write("std value: %f"%std_ratios + "\n\n")
    for ratio in ratios:
        txtmeanstd.write(str(ratio) + "\n")
    txtmeanstd.close()
    
### 得到SWratio
def get_SWtatio(result_dir_SW,SWratios_out,SWratios_tar,Diff_SWratios,SW_out,SW_tar):
    # 获取output图像的SWratio
    img_hsv2_S_out,img_hsv2_I_out,areaS_out,areaI_out = SW_out[0],SW_out[1],SW_out[2],SW_out[3]
    out_or_tar = "out"
    SWratio_out = StoIratio(result_dir_SW,img_hsv2_S_out,img_hsv2_I_out,areaS_out,areaI_out,out_or_tar)
    SWratios_out.append(SWratio_out)
    
    # 获取target图像的SWratio
    img_hsv2_S_tar,img_hsv2_I_tar,areaS_tar,areaI_tar = SW_tar[0],SW_tar[1],SW_tar[2],SW_tar[3]
    out_or_tar = "tar"
    SWratio_tar = StoIratio(result_dir_SW,img_hsv2_S_tar,img_hsv2_I_tar,areaS_tar,areaI_tar,out_or_tar)
    SWratios_tar.append(SWratio_tar)
    
    # 获取二者图像的SWratio差异
    Diff_SWratio = abs((SWratio_out-SWratio_tar)/SWratio_out)
    Diff_SWratios.append(Diff_SWratio)
    
    return SWratios_out,SWratios_tar,Diff_SWratios

### 运行主函数
def main_WIoU_SWratio(num_cases,img_path,p2p_or_p2pHD,results_path):
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
        
    MIoUs,WIoUs,PAs = [],[],[]
    SWratios_out,SWratios_tar,Diff_SWratios = [],[],[]
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

        if os.path.exists(imgname_target) and os.path.exists(imgname_output):  
            ## 图像数据读入与格式转化
            # 图像变为RGB矩阵
            imageR = imgname_target
            imageP = imgname_output
            arrayR0 = cv2.resize(cv2.imread(imageR),img_size) #读入图像数据,RGB,255,并统一大小
            arrayP0 = cv2.resize(cv2.imread(imageP),img_size) #读入图像数据,RGB,255,并统一大小
            # RGB矩阵变为HSV矩阵
            arrayR1 = cv2.cvtColor(arrayR0, cv2.COLOR_BGR2HSV)
            arrayP1 = cv2.cvtColor(arrayP0, cv2.COLOR_BGR2HSV)
            
            # 根据HSV的范围判断每个像素的类别
            arrayR2,array_newS_tar,array_newI_tar,pixel_numS_tar,pixel_numI_tar = switch_image(arrayR1)
            SW_tar = [array_newS_tar,array_newI_tar,pixel_numS_tar,pixel_numI_tar]
            
            arrayP2,array_newS_out,array_newI_out,pixel_numS_out,pixel_numI_out = switch_image(arrayP1)
            SW_out = [array_newS_out,array_newI_out,pixel_numS_out,pixel_numI_out]
            
            # 根据像素分类结果计算WIoU
            root_imgpath_WIoU = os.path.join(results_path,"WIoU")
            result_dir_WIoU = os.path.join(root_imgpath_WIoU,"test(%d)"%(img_NO+1)) #创建结果文件夹
            if not os.path.exists(result_dir_WIoU):
                os.makedirs(result_dir_WIoU)
            MIoUs,WIoUs,PAs = get_WIoU(arrayR2,arrayP2,imgname_target,imgname_output,img_size,result_dir_WIoU,img_NO,MIoUs,WIoUs,PAs)
            
            # 根据像素分类结果计算SWratio
            root_imgpath_SW = os.path.join(results_path,"SWratio")
            result_dir_SW = os.path.join(root_imgpath_SW,"test(%d)"%(img_NO+1)) #创建结果文件夹
            if not os.path.exists(result_dir_SW):
                os.makedirs(result_dir_SW)
            SWratios_out,SWratios_tar,Diff_SWratios = get_SWtatio(result_dir_SW,SWratios_out,SWratios_tar,Diff_SWratios,SW_out,SW_tar)
    
    try:    
        SWratios_out_path = os.path.join(root_imgpath_SW,"SWratios_out.txt")
        SWratios_tar_path = os.path.join(root_imgpath_SW,"SWratios_tar.txt")
        Diff_SWratios_path = os.path.join(root_imgpath_SW,"Diff_SWratios.txt")
        SWoutput_txt(SWratios_out,SWratios_out_path)
        SWoutput_txt(SWratios_tar,SWratios_tar_path)
        SWoutput_txt(Diff_SWratios,Diff_SWratios_path)
    except:
        print ("No SWratios")

    return MIoUs,WIoUs,PAs, Diff_SWratios,SWratios_out,SWratios_tar
