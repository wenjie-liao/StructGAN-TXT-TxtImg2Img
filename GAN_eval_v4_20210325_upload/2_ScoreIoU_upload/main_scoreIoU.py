# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 09:58:55 2020

@author: Administrator
"""

import sub1_rename_images as re_img
import sub2_divide_images as div_img
import sub3_SIoU as get_SIoU
import sub4_WIoU_SWratio as get_WIoU_SWratio
import sub5_meanstd as get_meanstd
import os
import numpy as np
import time

def get_score_iou(SIoUs,WIoUs,Diff_SWratios,score_iou_txtpath):
    score_ious = []
    w_SIoU,w_WIoU = 0.5,0.5
    with open(score_iou_txtpath,"w") as f_scoreiou:
        for i,Diff_SWratio in enumerate(Diff_SWratios):
            SIoU,WIoU = SIoUs[i],WIoUs[i]
            w_SWratio = (1-Diff_SWratio)
            score_iou = w_SWratio*(w_SIoU*SIoU + w_WIoU*WIoU)
            score_ious.append(score_iou)
            f_scoreiou.write(str(score_iou)+"\n")
        
    return score_ious  

def main_score():
    img_path = ".\\GANs\\images"
    results_path = ".\\GANs\\results"
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    num_cases = 2 #需要评价的案例的数量
    p2p_or_p2pHD = "p2pHD" #评价的对象是p2p的结果还是p2pHD的结果，["p2p","p2pHD","ManiGAN"]
    
    S_time = time.time()
    ### 对所有的图像格式进行改变
    print("1: rename all figures as png")
    re_img.filerename(img_path,'jpg','png')
    
    ### 将所有的图像划分子图，便于后续的轮廓提取
    print("2: subdivide all figures as sub-images")
    div_img.main_div_img(num_cases,img_path,p2p_or_p2pHD)
    
    ### 计算SIoU
    print("3: get SIoU")
    SIoUs = get_SIoU.main_SIoU(num_cases,img_path,p2p_or_p2pHD,results_path)
    
    ### 计算WIoU,MIoU,PA,SWratio的差异
    print("4: get WIoU, difference of SWratio")
    MIoUs,WIoUs,PAs, Diff_SWratios,SWratios_out,SWratios_tar = get_WIoU_SWratio.main_WIoU_SWratio(num_cases,img_path,p2p_or_p2pHD,results_path)

    ### 计算ScoreIoU
    print("5: get ScoreIoU")
    score_iou_txtpath = os.path.join(results_path,"score_iou.txt")
    score_ious = get_score_iou(SIoUs,WIoUs,Diff_SWratios,score_iou_txtpath)
    
    ### 输出SIOU,WIOU,MIOU等参数的均值和标准差
    get_meanstd.main_meanstd(num_cases,results_path)
    
    E_time = time.time()
    time_cost = E_time-S_time
    print("time cost: %d s"%time_cost)
    
    return None
    
### 执行主函数
main_score()