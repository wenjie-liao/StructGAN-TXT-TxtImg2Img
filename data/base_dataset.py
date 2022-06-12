import torch
import torch.utils.data as data
from PIL import Image
from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
import torchvision.transforms as transforms
import numpy as np
import random
import os

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.resize_or_crop == 'resize_and_crop':
        new_h = new_w = opt.loadSize            
    elif opt.resize_or_crop == 'scale_width_and_crop':
        new_w = opt.loadSize
        new_h = opt.loadSize * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.fineSize))
    y = random.randint(0, np.maximum(0, new_h - opt.fineSize))
    
    flip = random.random() > 0.5
    return {'crop_pos': (x, y), 'flip': flip}

def get_transform(opt, params, method=Image.BICUBIC, normalize=True):
    transform_list = []
    if 'resize' in opt.resize_or_crop:
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Scale(osize, method))   
    elif 'scale_width' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.loadSize, method)))
        
    if 'crop' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.fineSize)))

    if opt.resize_or_crop == 'none':
        base = float(2 ** opt.n_downsample_global)
        if opt.netG == 'local':
            base *= (2 ** opt.n_local_enhancers)
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base, method)))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def get_trans_txt(opt,txt):
#    txt_expand = np.tile(txt,(opt.n_captions,opt.n_maxpersen))
    txt_expand = txt
    tensor_txt = torch.tensor(txt_expand,dtype=torch.float32)
    return tensor_txt   

def normalize():    
    return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size        
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)

def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img    
    w = target_width
    h = int(target_width * oh / ow)    
    return img.resize((w, h), method)

def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):        
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img

def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img

class CreateTextDataset():
    def __init__(self, opt):
        self.opt = opt
        self.rawtxt_dir = self.opt.rawtxt_dir
        self.train_txt_dir = os.path.join(self.opt.dataroot,"train_txt")
        if not os.path.exists(self.train_txt_dir):
            os.makedirs(self.train_txt_dir)
        self.test_txt_dir = os.path.join(self.opt.dataroot,"test_txt")
        if not os.path.exists(self.test_txt_dir):
            os.makedirs(self.test_txt_dir)

    def load_captions(self,data_dir,filenames):
        all_captions = []
        for i,filename in enumerate(filenames):
            cap_path = os.path.join(data_dir,filename)
            with open(cap_path, "r") as f:
                captions = f.read().split('\n')
                cnt = 0
                for cap in captions:
                    if len(cap) == 0:
                        continue
                    cap = cap.replace("\ufffd\ufffd", " ")
                    # picks out sequences of alphanumeric characters as tokens
                    # and drops everything else
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(cap.lower())
    
                    if len(tokens) == 0:
                        print('cap', cap)
                        continue
    
                    tokens_new = []
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        if len(t) > 0:
                            tokens_new.append(t)
                    all_captions.append({filename:{cnt:tokens_new}})
                    cnt += 1
                    if cnt == self.opt.n_captions:
                        break
                if cnt < self.opt.n_captions:
                    print('ERROR: the captions for %s less than %d'
                          % (filenames[i], cnt))
        return all_captions

    def build_dictionary(self,train_captions, test_captions):
        ixtoword = {}
        ixtoword[0] = '<end>'
        wordtoix = {}
        wordtoix['<end>'] = 0
        for cnt in range(self.opt.n_captions):
            word_counts = defaultdict(float)
            captions = train_captions + test_captions
            for sent in captions:
                for key,words_dict in sent.items():
                    for key,words in words_dict.items():
                        if key == cnt:
                            for word in words:
                                word_counts[word] += 1
            vocab = [w for w in word_counts if word_counts[w] >= 0]
            ix = 5.0
            for w in vocab:
                wordtoix[w] = ix + cnt*self.opt.max_n_prop
                ixtoword[ix] = w
                ix += 5.0
    
        train_captions_new = []
        for t in train_captions:
            rev,temp_rev = [],[]
            for keytrain,ws_dict in t.items():
                for keycnt,ws in ws_dict.items():
                    for w in ws:
                        if w in wordtoix:
                            rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            # this train_captions_new hold index of each word in sentence
            while len(rev) < self.opt.n_maxpersen:
                rev = rev + rev
            if len(rev) > self.opt.n_maxpersen:
                temp_rev = rev[:self.opt.n_maxpersen]
            else:
                temp_rev = rev
            train_captions_new.append({keytrain:temp_rev})
    
        test_captions_new = []
        for t in test_captions:
            rev,temp_rev = [],[]
            for keytest,ws_dict in t.items():
                for keycnt,ws in ws_dict.items():
                    for w in ws:
                        if w in wordtoix:
                            rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            while len(rev) < self.opt.n_maxpersen:
                rev = rev + rev
            if len(rev) > self.opt.n_maxpersen:
                temp_rev = rev[:self.opt.n_maxpersen]
            else:
                temp_rev = rev
            test_captions_new.append({keytest:temp_rev})
    
        return [train_captions_new, test_captions_new,
                ixtoword, wordtoix, len(ixtoword)]

    def load_text_data(self):
        train_captions,test_captions = [],[]
        #obtain train captions
        train_names = os.listdir(os.path.join(self.rawtxt_dir,"train"))
        train_captions = self.load_captions(os.path.join(self.rawtxt_dir,"train"), train_names)
        #obtain test captions
        test_names = os.listdir(os.path.join(self.rawtxt_dir,"test"))
        test_captions = self.load_captions(os.path.join(self.rawtxt_dir,"test"), test_names)
        #transfer Words list to number list
        train_captions, test_captions, ixtoword, wordtoix, n_words = self.build_dictionary(train_captions, test_captions)
            
        # combine train and test datasets
        captions = train_captions + test_captions
        filenames = train_names + test_names
    
        return filenames,train_names,test_names,captions,train_captions,test_captions, ixtoword, wordtoix, n_words
    
    def txtcode(self):
        #load text
        txtnames,train_names,test_names,captions,train_captions,test_captions, ixtoword, wordtoix, n_words = self.load_text_data()
        #write new captions in text
        for i,train_caption in enumerate(train_captions):
            if i!= 0:
                pre_train_txtname = [pre_train_txtname for pre_train_txtname in train_captions[i-1].keys()][0]
                train_txtname = [train_txtname for train_txtname in train_caption.keys()][0]
                caps = [caps for caps in train_caption.values()][0]
                if train_txtname != pre_train_txtname:
                    train_txt_outpath = os.path.join(self.train_txt_dir,train_txtname)
                    with open(train_txt_outpath, 'w') as train_txt_out:
                        for i,cap in enumerate(caps):
                            if i < (len(caps)-1):
                                train_txt_out.write(str(cap)+"\t")
                            else:
                                train_txt_out.write(str(cap)+"\n")
                else:
                    with open(train_txt_outpath, 'a+') as train_txt_out:
                        for i,cap in enumerate(caps):
                            if i < (len(caps)-1):
                                train_txt_out.write(str(cap)+"\t")
                            else:
                                train_txt_out.write(str(cap)+"\n")
            else:
                train_txtname = [train_txtname for train_txtname in train_caption.keys()][0]
                caps = [caps for caps in train_caption.values()][0]
                train_txt_outpath = os.path.join(self.train_txt_dir,train_txtname)
                with open(train_txt_outpath, 'w') as train_txt_out:
                    for i,cap in enumerate(caps):
                        if i < (len(caps)-1):
                            train_txt_out.write(str(cap)+"\t")
                        else:
                            train_txt_out.write(str(cap)+"\n")
                            
        for i,test_caption in enumerate(test_captions):
            if i!= 0:
                pre_test_txtname = [pre_test_txtname for pre_test_txtname in test_captions[i-1].keys()][0]
                test_txtname = [test_txtname for test_txtname in test_caption.keys()][0]
                caps = [caps for caps in test_caption.values()][0]
                if test_txtname != pre_test_txtname:
                    test_txt_outpath = os.path.join(self.test_txt_dir,test_txtname)
                    with open(test_txt_outpath, 'w') as test_txt_out:
                        for i,cap in enumerate(caps):
                            if i < (len(caps)-1):
                                test_txt_out.write(str(cap)+"\t")
                            else:
                                test_txt_out.write(str(cap)+"\n")
                else:
                    with open(test_txt_outpath, 'a+') as test_txt_out:
                        for i,cap in enumerate(caps):
                            if i < (len(caps)-1):
                                test_txt_out.write(str(cap)+"\t")
                            else:
                                test_txt_out.write(str(cap)+"\n")
            else:
                test_txtname = [test_txtname for test_txtname in test_caption.keys()][0]
                caps = [caps for caps in test_caption.values()][0]
                test_txt_outpath = os.path.join(self.test_txt_dir,test_txtname)
                with open(test_txt_outpath, 'w') as test_txt_out:
                    for i,cap in enumerate(caps):
                        if i < (len(caps)-1):
                            test_txt_out.write(str(cap)+"\t")
                        else:
                            test_txt_out.write(str(cap)+"\n")
                
        return txtnames, captions