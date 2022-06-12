import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize, get_trans_txt
from data.image_folder import make_dataset
from data.text_folder import make_dataset_txt
from PIL import Image
import numpy as np
import copy

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    

        ### input A (label maps)
        dir_A = '_A' if self.opt.label_nc == 0 else '_label'
        self.dir_A = os.path.join(opt.dataroot, (opt.phase + dir_A))
        self.A_paths = sorted(make_dataset(self.dir_A))
        
        ### input txt (label txts)
        dir_txt = '_txt'
        self.dir_txt = os.path.join(opt.dataroot, (opt.phase + dir_txt))
        self.txt_paths = sorted(make_dataset_txt(self.dir_txt))

        ### input B (real images)
        if opt.isTrain or opt.use_encoded_image:
            dir_B = '_B' if self.opt.label_nc == 0 else '_img'
            self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)  
            self.B_paths = sorted(make_dataset(self.dir_B))

        ### instance maps
        if not opt.no_instance:
            self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
            self.inst_paths = sorted(make_dataset(self.dir_inst))

        ### load precomputed instance-wise encoded features
        if opt.load_features:                              
            self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
            print('----------- loading features from %s ----------' % self.dir_feat)
            self.feat_paths = sorted(make_dataset(self.dir_feat))

        self.dataset_size = len(self.A_paths) 
      
    def __getitem__(self, index):        
        ### input A (label maps)
        A_path = self.A_paths[index]              
        A = Image.open(A_path)        
        params = get_params(self.opt, A.size)
        if self.opt.label_nc == 0:
            transform_A = get_transform(self.opt, params)
            A_tensor = transform_A(A.convert('RGB'))
        else:
            transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            A_tensor = transform_A(A) * 255.0

        ### input txt (label txts)
        txt_path =self.txt_paths[index]
        txt = np.loadtxt(txt_path)
        txt_tensor = get_trans_txt(self.opt,txt)
        
        ### fake txt (fake txts)
        fake_txt = copy.deepcopy(txt)
        while (fake_txt==txt).all():
            rand_index = np.random.randint(0,len(self.txt_paths))
            fake_txt_path =self.txt_paths[rand_index]
            fake_txt = np.loadtxt(fake_txt_path)
            fake_txt_tensor = get_trans_txt(self.opt,fake_txt)
        
        ### input B (real images)
        B_tensor = inst_tensor = feat_tensor = 0
        if self.opt.isTrain or self.opt.use_encoded_image:
            B_path = self.B_paths[index]   
            B = Image.open(B_path).convert('RGB')
            transform_B = get_transform(self.opt, params)      
            B_tensor = transform_B(B)

        ### if using instance maps        
        if not self.opt.no_instance:
            inst_path = self.inst_paths[index]
            inst = Image.open(inst_path)
            inst_tensor = transform_A(inst)

            if self.opt.load_features:
                feat_path = self.feat_paths[index]            
                feat = Image.open(feat_path).convert('RGB')
                norm = normalize()
                feat_tensor = norm(transform_A(feat))                            

        input_dict = {'label': A_tensor,'label_txt': txt_tensor,'fake_txt': fake_txt_tensor ,'inst': inst_tensor, 
                      'image': B_tensor, 'feat': feat_tensor, 'path': A_path}

        return input_dict

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'