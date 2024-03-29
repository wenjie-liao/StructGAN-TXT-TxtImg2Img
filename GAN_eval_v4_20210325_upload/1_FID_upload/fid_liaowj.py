'''
From https://github.com/tsc2017/Frechet-Inception-Distance
Code derived from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/gan/python/eval/python/classifier_metrics_impl.py

Usage:
    Call get_fid(images1, images2)
Args:
    images1, images2: Numpy arrays with values ranging from 0 to 255 and shape in the form [N, 3, HEIGHT, WIDTH] where N, HEIGHT and WIDTH can be arbitrary. 
    dtype of the images is recommended to be np.uint8 to save CPU memory.
Returns:
    Frechet Inception Distance between the two image distributions.
'''

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
import functools
import numpy as np
import time
import shutil
import cv2
from tensorflow.python.ops import array_ops
# pip install tensorflow-gan
import tensorflow_gan as tfgan

#session=tf.compat.v1.InteractiveSession()
session=tf.InteractiveSession()
# A smaller BATCH_SIZE reduces GPU memory usage, but at the cost of a slight slowdown
BATCH_SIZE = 32

# Run images through Inception.
inception_images = tf.placeholder(tf.float32, [None, 3, None, None], name = 'inception_images')
activations1 = tf.placeholder(tf.float32, [None, None], name = 'activations1')
activations2 = tf.placeholder(tf.float32, [None, None], name = 'activations2')
fcd = tfgan.eval.frechet_classifier_distance_from_activations(activations1, activations2)

#INCEPTION_TFHUB = 'https://tfhub.dev/tensorflow/tfgan/eval/inception/1'
INCEPTION_TFHUB = "C:/Users/Admin/.conda/envs/tensor_v2/tfgan_eval_inception_1"
INCEPTION_FINAL_POOL = 'pool_3'

def inception_activations(images = inception_images, num_splits = 1):
    images = tf.transpose(images, [0, 2, 3, 1])
    size = 299
    images = tf.image.resize_bilinear(images, [size, size])
    generated_images_list = array_ops.split(images, num_or_size_splits = num_splits)
    activations = tf.map_fn(
        fn = tfgan.eval.classifier_fn_from_tfhub(INCEPTION_TFHUB, INCEPTION_FINAL_POOL, True),
        elems = array_ops.stack(generated_images_list),
        parallel_iterations = 1,
        back_prop = False,
        swap_memory = True,
        name = 'RunClassifier')
    activations = array_ops.concat(array_ops.unstack(activations), 0)
    return activations

activations =inception_activations()

def get_inception_activations(inps):
    n_batches = int(np.ceil(float(inps.shape[0]) / BATCH_SIZE))
    act = np.zeros([inps.shape[0], 2048], dtype = np.float32)
    for i in range(n_batches):
        inp = inps[i * BATCH_SIZE : (i + 1) * BATCH_SIZE] / 255. * 2 - 1
        act[i * BATCH_SIZE : i * BATCH_SIZE + min(BATCH_SIZE, inp.shape[0])] = session.run(activations, feed_dict = {inception_images: inp})
    return act

def activations2distance(act1, act2):
    return session.run(fcd, feed_dict = {activations1: act1, activations2: act2})
        
def get_fid(images1, images2):
    session=tf.get_default_session()
    assert(type(images1) == np.ndarray)
    assert(len(images1.shape) == 4)
    assert(images1.shape[1] == 3)
    assert(np.min(images1[0]) >= 0 and np.max(images1[0]) > 10), 'Image values should be in the range [0, 255]'
    assert(type(images2) == np.ndarray)
    assert(len(images2.shape) == 4)
    assert(images2.shape[1] == 3)
    assert(np.min(images2[0]) >= 0 and np.max(images2[0]) > 10), 'Image values should be in the range [0, 255]'
    assert(images1.shape == images2.shape), 'The two numpy arrays must have the same shape'
    print('Calculating FID with %i images from each distribution' % (images1.shape[0]))
    start_time = time.time()
    act1 = get_inception_activations(images1)
    act2 = get_inception_activations(images2)
    fid = activations2distance(act1, act2)
    print('FID calculation time: %f s' % (time.time() - start_time))
    return fid

def filerename(gen_path):
    genimg_names = os.listdir(gen_path)
    for genimg_name in genimg_names:
        if genimg_name[-16:] == "_input_label.png":
            input_lable = os.path.join(gen_path,genimg_name)
            os.remove (input_lable)
        elif genimg_name[-22:] == "_synthesized_image.png":
            synthesized_image = os.path.join(gen_path,genimg_name)
            synthesized_image_new = os.path.join(gen_path,(genimg_name[:-22]+".png"))
            os.rename (synthesized_image,synthesized_image_new)

def fileread(root_path):
    tar_path = os.path.join(root_path,"target")
    gen_path = os.path.join(root_path,"images")
    
    filerename(gen_path)
    
    tarimg_names = os.listdir(tar_path)
    genimg_names = os.listdir(gen_path)
    
    tarimgs,genimgs = [],[]
    for i,tarimg_name in enumerate(tarimg_names):
        tarimg = cv2.imread(os.path.join(tar_path,tarimg_name))
#        tarimg = cv2.resize(tarimg,(256,512))
        tarimg = np.transpose(tarimg,(2,0,1))
#        tarimg = tarimg[np.newaxis,:]
        tarimgs.append(tarimg)
        genimg = cv2.imread(os.path.join(gen_path,genimg_names[i]))
#        genimg = cv2.resize(genimg,(256,512))
        genimg = np.transpose(genimg,(2,0,1))
#        genimg = genimg[np.newaxis,:]
        genimgs.append(genimg)
        
    tarimgs = np.array(tarimgs)
    genimgs = np.array(genimgs)
        
    return tarimgs,genimgs

def ManiGANcatch(root_path):
    Main_path = os.path.join(root_path,"bird_Main")
    images_path = os.path.join(root_path,"images")
    if not os.path.exists(images_path):
        os.makedirs(images_path)
    main_dir_names = os.listdir(Main_path)
    for main_dir_name in main_dir_names:
        main_img_path = os.path.join(Main_path,main_dir_name,"1_sf_3_SF.png")
        images_img_path = os.path.join(images_path,(main_dir_name+".png"))
        shutil.copyfile(main_img_path,images_img_path)
        
    return None

def fileretype(filepath,srctype,destype):
    for path,dirlist,filelist in os.walk(filepath):
        for file in filelist:

            #防止文件名中包含.
            fullist = file.split('.')
            namelist = fullist[0:-1]
            filename = ''
            for i in namelist:
                filename = filename + i + '.' 
            # print (filename)

            curndir = os.getcwd()    #获取当前路径
            # print (curndir)

            os.chdir(path)            #设置当前路径为目标目录
            newdir = os.getcwd()    #验证当前目录
            # print (newdir)

            filetype = file.split('.')[-1]    #获取目标文件格式

            if filetype == srctype:    #修改目标目录下指定后缀的文件（包含子目录）
                os.rename(file,filename+destype)

            if srctype == '*':        #修改目标目录下所有文件后缀（包含子目录）
                os.rename(file,filename+destype)

            if srctype == 'null':    #修改目标目录下所有无后缀文件（包含子目录）
                if len(fullist) == 1:
                    os.rename(file,file+'.'+destype)

            os.chdir(curndir)    #回到之前的路径
            
    return None

def main():
    root_path = ".\\structs"
#    ManiGANcatch(root_path)
    fileretype(root_path,'jpg','png')
    
    tarimgs,genimgs = fileread(root_path)
    fid = get_fid(tarimgs,genimgs)
    
#    fids = []
#    for i,tarimg in enumerate(tarimgs):
#        genimg = genimgs[i]
#        fid = get_fid(tarimg,genimg)
#        fids.append(fid)

    fid = np.array(fid)
    results_fid = os.path.join(root_path,"results_fid.txt")
    with open(results_fid,"w") as results_fid_w:
        results_fid_w.write(str(fid))
    
main()