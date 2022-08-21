# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 15:06:57 2020

@author: 12437
"""
import numpy as np
import os

def read_SIoUs(num_cases,path):
    SIoUs = []
    for img_NO in range(num_cases):
        result_dir = os.path.join(path,"test(%d)"%(img_NO+1))
        if os.path.exists(result_dir): #判断结果文件夹是否存在
            try:
                IoU_path = os.path.join(result_dir,"SIoU(%d).txt"%(img_NO+1))
                txttemp = open(IoU_path,"r")
            except:
                IoU_path = os.path.join(result_dir,"IoU(%d).txt"%(img_NO+1))
                txttemp = open(IoU_path,"r")
            lines = txttemp.readlines()
            for i,line in enumerate(lines):
                if i==0:
                    templine = line.split(":")[-1]
                    SIoUs.append(float(templine))
                    
    mean_SIoUs = np.mean(SIoUs)
    std_SIoUs = np.std(SIoUs)
    
    return SIoUs,mean_SIoUs,std_SIoUs

def read_WIoUs(num_cases,path):
    MIoUs,WIoUs,PAs = [],[],[]
    for img_NO in range(num_cases):
        result_dir = os.path.join(path,"test(%d)"%(img_NO+1))
        if os.path.exists(result_dir): #判断结果文件夹是否存在
            IoU_path = os.path.join(result_dir,"pixIoU(%d).txt"%(img_NO+1))
            txttemp = open(IoU_path,"r")
            lines = txttemp.readlines()
            for i,line in enumerate(lines):
                if i==0:
                    templine = line.split(":")[-1]
                    MIoUs.append(float(templine))
                elif i==1:
                    templine = line.split(":")[-1]
                    WIoUs.append(float(templine))
                elif i==2:
                    templine = line.split(":")[-1]
                    PAs.append(float(templine))
                    
    mean_MIoU = np.mean(MIoUs)
    std_MIoU = np.std(MIoUs)
    mean_WIoU = np.mean(WIoUs)
    std_WIoU = np.std(WIoUs)
    mean_PAs = np.mean(PAs)
    std_PAs = np.std(PAs)
    
    return MIoUs,mean_MIoU,std_MIoU,WIoUs,mean_WIoU,std_WIoU,PAs,mean_PAs,std_PAs

def output_txt(IoUs,mean,std,txtpath):
    txtmeanstd=open(txtpath,"w")
    txtmeanstd.write("mean value: %f"%mean + "\n")
    txtmeanstd.write("std value: %f"%std + "\n\n")
    for subIoU in IoUs:
        txtmeanstd.write(str(subIoU) + "\n")
    txtmeanstd.close()
    
## 运行函数
def main_meanstd(num_cases,results_path):
    SIoU_path = os.path.join(results_path,"SIoU")
    WIoU_path = os.path.join(results_path,"WIoU")

    SIoUs,mean_SIoU,std_SIoU = read_SIoUs(num_cases,SIoU_path)
    MIoUs,mean_MIoU,std_MIoU,WIoUs,mean_WIoU,std_WIoU,PAs,mean_PAs,std_PAs = read_WIoUs(num_cases,WIoU_path)
    SIoUpath = os.path.join(results_path,"SIoUmean_std.txt")
    MIoUpath = os.path.join(results_path,"MIoUmean_std.txt")
    WIoUpath = os.path.join(results_path,"WIoUmean_std.txt")
    PAspath = os.path.join(results_path,"PAsmean_std.txt")
    output_txt(SIoUs,mean_SIoU,std_SIoU,SIoUpath)
    output_txt(MIoUs,mean_MIoU,std_MIoU,MIoUpath)
    output_txt(WIoUs,mean_WIoU,std_WIoU,WIoUpath)
    output_txt(PAs,mean_PAs,std_PAs,PAspath)
    
    return None
