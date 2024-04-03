import cv2
import glob
import json
import numpy as np
import os, sys, traceback
import pandas as pd
import random
import time
import torch
import decord
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as trans

import config as cfg


decord.bridge.set_bridge('torch')


# Validation dataset.
class baseline_val_dataloader(Dataset):

    def __init__(self, video_list=None, model_name='vjepa', dataset='ucf101', shuffle=True, data_percentage=1.0, mode=0, total_num_modes=1, skip_rate=1, num_frames=16):
        
        self.total_num_modes = total_num_modes
        if self.total_num_modes == 1:
            self.total_num_modes = 5
            self.mode = 2
        self.skip_rate = skip_rate
        self.num_frames = num_frames
        self.model_name = model_name

        self.dataset = dataset if video_list is None else 'custom'
        self.all_paths = video_list

        if video_list is None:
            split = 1
            if self.dataset == 'ucf101' or self.dataset == 'ucf101_dino':
                self.classes = json.load(open(cfg.ucf101_class_mapping))['classes']
                if split <= 3:
                    all_paths = open(os.path.join(cfg.ucf101_path, 'ucfTrainTestlist', f'testlist0{split}.txt'),'r').read().splitlines()
                    self.all_paths = [x.replace('/', os.sep) for x in all_paths]
                else:
                    print(f'Invalid split input: {split}')    
            
            elif self.dataset == 'hmdb51':
                anno_file = pd.read_csv(os.path.join(cfg.hmdb_file_path, 'hmdb51_test_labels.csv'), index_col=None)
                self.all_paths = [f'{os.path.join(cfg.hmdb_vid_path, c, f)} {l}' for c, f, l in zip(anno_file['class'].to_list(), anno_file['filename'].to_list(), anno_file['label'].to_list())]

            elif self.dataset == 'pahmdb':
                anno_file = pd.read_csv(os.path.join(cfg.hmdb_file_path, 'pahmdb_labels.csv'), index_col=None)
                self.all_paths = [f'{os.path.join(cfg.hmdb_vid_path, c, f)} {l}' for c, f, l in zip(anno_file['class'].to_list(), anno_file['filename'].to_list(), anno_file['label'].to_list())]
                    
            elif self.dataset == 'k400':
                # self.all_paths = open(os.path.join(cfg.kinetics_path, 'k400val_full_path_resized_annos.txt'),'r').read().splitlines()
                all_paths = open(os.path.join(cfg.kinetics_path, 'annotation_test_fullpath_resizedvids.txt'),'r').read().splitlines()
                self.all_paths = [x.replace(f'/home/c3-0/datasets/kin400_resized/test/', os.path.join(cfg.kinetics_path, 'test') + os.sep) for x in all_paths]

            elif self.dataset == 'ucf101_obf':
                self.all_paths = glob.glob('/home/c3-0/ishan/privacy_preserving1/yolo_based_obfuscation/UCF101/blackened/test/*/*')
                self.classes = json.load(open(cfg.ucf101_class_mapping))['classes']

            elif self.dataset == 'cap':
                # self.classes = json.load(open(os.path.join(cfg.cap_path, 'val_list.json')))
                classes = pd.read_csv(os.path.join(cfg.cap_path, 'val_list.csv'), index_col=0).to_dict('index')
                self.classes = {k: (v['label'], v['superlabel']) for k, v in classes.items()}
                self.all_paths = list(self.classes.keys())

            elif self.dataset == 'aras':
                classes = pd.read_csv(os.path.join(cfg.bias_path, 'ARAS', 'annotations', 'aras_annotations.csv'), index_col=0)
                self.all_paths = classes.index.to_list()
                self.classes = classes['label'].to_list()
                self.all_paths = [(x, y) for x, y in zip(self.all_paths, self.classes)]

            elif self.dataset == 'mimetics':
                classes = pd.read_csv(os.path.join(cfg.bias_path, 'mimetics', 'mimetics_labels_updated.csv'))
                self.all_paths = classes['filename'].to_list()
                self.classes = classes['label'].to_list()
                self.all_paths = [(x, y) for x, y in zip(self.all_paths, self.classes)]

            elif self.dataset == 'ucf101_scuba_places365':
                self.classes = json.load(open(cfg.ucf101_class_mapping))['classes']
                self.all_paths = sorted(glob.glob(os.path.join(cfg.bias_path, 'UCF101_SCUBA_Places365', '*'), recursive=False))
            
            elif self.dataset == 'ucf101_scuba_stripe':
                self.classes = json.load(open(cfg.ucf101_class_mapping))['classes']
                self.all_paths = sorted(glob.glob(os.path.join(cfg.bias_path, 'UCF101_SCUBA_Stripe', '*'), recursive=False))

            elif self.dataset == 'ucf101_scuba_vqgan':
                self.classes = json.load(open(cfg.ucf101_class_mapping))['classes']
                self.all_paths = sorted(glob.glob(os.path.join(cfg.bias_path, 'UCF101_SCUBA_VQGAN', '*'), recursive=False))
    
            elif self.dataset == 'ucf101_conflfg_stripe':
                self.classes = json.load(open(cfg.ucf101_class_mapping))['classes']
                self.all_paths = sorted(glob.glob(os.path.join(cfg.bias_path, 'UCF101_ConflFG_Stripe', '*'), recursive=False))

            elif self.dataset == 'ucf101_scufo_places365':
                self.classes = json.load(open(cfg.ucf101_class_mapping))['classes']
                self.all_paths = sorted(glob.glob(os.path.join(cfg.bias_path, 'UCF101_SCUFO_Places365', '*'), recursive=False))
            
            elif self.dataset == 'ucf101_scufo_stripe':
                self.classes = json.load(open(cfg.ucf101_class_mapping))['classes']
                self.all_paths = sorted(glob.glob(os.path.join(cfg.bias_path, 'UCF101_SCUFO_Stripe', '*'), recursive=False))

            elif self.dataset == 'ucf101_scufo_vqgan':
                self.classes = json.load(open(cfg.ucf101_class_mapping))['classes']
                self.all_paths = sorted(glob.glob(os.path.join(cfg.bias_path, 'UCF101_SCUFO_VQGAN', '*'), recursive=False))

            elif self.dataset == 'hmdb51_scuba_places365':
                anno_file = pd.read_csv(os.path.join(cfg.hmdb_file_path, 'hmdb51_labels.csv'), index_col=None)
                self.classes = {os.path.basename(k): v for k, v in zip(anno_file['filename'].to_list(), anno_file['label'].to_list())}
                self.all_paths = sorted(glob.glob(os.path.join(cfg.bias_path, 'HMDB51_SCUBA_Places365', '*'), recursive=False))
            
            elif self.dataset == 'hmdb51_scuba_stripe':
                anno_file = pd.read_csv(os.path.join(cfg.hmdb_file_path, 'hmdb51_labels.csv'), index_col=None)
                self.classes = {os.path.basename(k): v for k, v in zip(anno_file['filename'].to_list(), anno_file['label'].to_list())}
                self.all_paths = sorted(glob.glob(os.path.join(cfg.bias_path, 'HMDB51_SCUBA_Stripe', '*'), recursive=False))

            elif self.dataset == 'hmdb51_scuba_vqgan':
                anno_file = pd.read_csv(os.path.join(cfg.hmdb_file_path, 'hmdb51_labels.csv'), index_col=None)
                self.classes = {os.path.basename(k): v for k, v in zip(anno_file['filename'].to_list(), anno_file['label'].to_list())}
                self.all_paths = sorted(glob.glob(os.path.join(cfg.bias_path, 'HMDB51_SCUBA_VQGAN', '*'), recursive=False))

            elif self.dataset == 'hmdb51_conflfg_stripe':
                anno_file = pd.read_csv(os.path.join(cfg.hmdb_file_path, 'hmdb51_labels.csv'), index_col=None)
                self.classes = {os.path.basename(k): v for k, v in zip(anno_file['filename'].to_list(), anno_file['label'].to_list())}
                self.all_paths = sorted(glob.glob(os.path.join(cfg.bias_path, 'HMDB51_ConflFG_Stripe', '*'), recursive=False))

            elif self.dataset == 'hmdb51_scufo_places365':
                anno_file = pd.read_csv(os.path.join(cfg.hmdb_file_path, 'hmdb51_labels.csv'), index_col=None)
                self.classes = {os.path.basename(k): v for k, v in zip(anno_file['filename'].to_list(), anno_file['label'].to_list())}
                self.all_paths = sorted(glob.glob(os.path.join(cfg.bias_path, 'HMDB51_SCUFO_Places365', '*'), recursive=False))

            elif self.dataset == 'hmdb51_scufo_stripe':
                anno_file = pd.read_csv(os.path.join(cfg.hmdb_file_path, 'hmdb51_labels.csv'), index_col=None)
                self.classes = {os.path.basename(k): v for k, v in zip(anno_file['filename'].to_list(), anno_file['label'].to_list())}
                self.all_paths = sorted(glob.glob(os.path.join(cfg.bias_path, 'HMDB51_SCUFO_Stripe', '*'), recursive=False))

            elif self.dataset == 'hmdb51_scufo_vqgan':
                anno_file = pd.read_csv(os.path.join(cfg.hmdb_file_path, 'hmdb51_labels.csv'), index_col=None)
                self.classes = {os.path.basename(k): v for k, v in zip(anno_file['filename'].to_list(), anno_file['label'].to_list())}
                self.all_paths = sorted(glob.glob(os.path.join(cfg.bias_path, 'HMDB51_SCUFO_VQGAN', '*'), recursive=False))

            elif self.dataset == 'TSH':
                anno_file = pd.read_csv(os.path.join(cfg.tsh_path, 'tsh_labels.csv'), index_col=None)
                # if self.params.split == 'male':
                #     anno_file = anno_file[anno_file['gender'] == 'f']
                # else:
                #     anno_file = anno_file[anno_file['gender'] == 'm']
                anno_file = anno_file[anno_file['split'] == 'test']

                self.all_paths = anno_file['filename'].to_list()
                self.classes = json.load(open(os.path.join(cfg.tsh_path, 'tsh_class_mapping.json'), 'r'))

            else:
                print(f'{self.dataset} does not exist.')
                
        self.shuffle = shuffle

        if self.shuffle:
            random.shuffle(self.all_paths)
        
        self.data_percentage = data_percentage
        self.data_limit = int(len(self.all_paths)*self.data_percentage)
        self.data = self.all_paths[0: self.data_limit]
        self.mode = mode

        if model_name == 'vjepa':
            self.augmentation = trans.Compose([
                trans.Resize(size=224, interpolation=trans.InterpolationMode.BILINEAR, antialias=False),
                trans.CenterCrop(size=(224, 224)),
                trans.ToPILImage(),
                trans.ToTensor(),  # Converts to tensor & scales to [0,1]
                trans.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif model_name == 'clip':
            self.augmentation = trans.Compose([
                trans.Lambda(lambda x: x / 255.0),
                trans.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
                trans.Resize(size=224, antialias=False),
                trans.CenterCrop(size=224),
                # trans.RandomHorizontalFlip(p=0.5)
            ])
    
    def __len__(self):
        return len(self.data)
            
    def __getitem__(self,index):        
        clip, label, vid_path, frame_list = self.process_data(index)
        return clip, label, vid_path, frame_list


    def process_data(self, idx):
        if self.dataset != 'custom':
            # Label building.
            if self.dataset == 'ucf101':
                vid_path1 = os.path.join(cfg.ucf101_path, 'Videos', self.data[idx].split(' ')[0])
                label = int(self.classes[vid_path1.split(os.sep)[-2]]) - 1 # This element should be activity name.
            elif self.dataset == 'ucf101_dino':
                vid_path1 = os.path.join(cfg.ucf101_path, 'dino_att', 'Videos', self.data[idx].split(' ')[0])
                label = int(self.classes[vid_path1.split(os.sep)[-2]]) - 1  # This element should be activity name.
            elif self.dataset == 'hmdb51' or self.dataset == 'pahmdb':
                vid_path1, label = self.data[idx].split(' ')
                label = int(label)
            elif self.dataset == 'k400':
                vid_path1 = self.data[idx].split(' ')[0]
                label = int(self.data[idx].split(' ')[1]) - 1 

            elif self.dataset == 'ucf101_obf':
                vid_path1 = self.data[idx]
                label = int(self.classes[vid_path1.split(os.sep)[-2]]) - 1 # This element should be activity name.

            elif self.dataset == 'cap':
                vid_name = self.data[idx]
                _, label  = self.classes[vid_name] # label, superlabel
                vid_path1 = os.path.join(cfg.cap_path, 'videos', vid_name)
            
            elif self.dataset == 'mimetics':
                vid_name, label = self.data[idx]
                vid_path1 = os.path.join(cfg.bias_path, 'mimetics', 'videos', vid_name)
                clip, frame_list = self.build_clip_mimetics(vid_path1)
                return clip, label, frame_list, idx

            elif self.dataset == 'aras':
                vid_name, label = self.data[idx]
                vid_path1 = os.path.join(cfg.bias_path, 'ARAS', vid_name)

            elif 'scuba' in self.dataset or 'conflfg' in self.dataset or 'scufo' in self.dataset:
                vid_path1 = self.data[idx]
                if 'ucf101' in self.dataset:
                    label = int(self.classes[vid_path1.split(f'{os.sep}v_')[-1].split('_g0')[0]]) - 1
                elif 'hmdb51' in self.dataset:
                    key = os.path.basename(vid_path1)[:-3] if 'scufo' in self.dataset else os.path.basename(vid_path1)[:-7]
                    label = self.classes[key + '.avi']
                else:
                    label = None  # TODO: add label for other datasets
                clip, frame_list = self.build_clip_scuba(vid_path1)
                return clip, label, frame_list, idx
            
            elif self.dataset == 'NTU':
                vid_path1 = self.data[idx]
                label = int(os.path.basename(vid_path1)[17:20]) - 1

            elif self.dataset == 'TSH':
                vid_path1 = os.path.join(cfg.tsh_path, 'videos', self.data[idx])
                label = self.classes[os.path.basename(vid_path1).split('_')[0]] - 1
        else:
            vid_path1 = self.data[idx]
            label = 0
        # Clip building.
        clip, frame_list = self.build_clip(vid_path1)
        return clip, label, frame_list, os.path.basename(vid_path1)

    
    def build_clip_scuba(self, vid_path):
        try:
            frame_list = sorted(glob.glob(os.path.join(vid_path, '*.jpg')))
            if 'scufo' in self.dataset:
                frame = torchvision.io.read_image(frame_list[0])
                _, self.ori_reso_h, self.ori_reso_w = frame.shape
                self.min_size = min(self.ori_reso_h, self.ori_reso_w)
                frame = self.augmentation(frame)
                list_full = [0 for _ in range(self.params.num_frames)]
                full_clip = []
                for i in range(self.params.num_frames):
                    full_clip.append(frame)
            else:
                # return self.build_clip(vid_path)
                if len(frame_list) == 0:
                    frame_list = sorted(glob.glob(os.path.join(vid_path.replace(cfg.bias_path, cfg.bias_frame_path)[:-4], '*.jpg')))

                frame_count = len(frame_list)
                skip_frames_full = self.params.fix_skip 

                if skip_frames_full*self.params.num_frames > frame_count:
                    skip_frames_full /= 2

                left_over = skip_frames_full*self.params.num_frames
                F = frame_count - left_over

                start_frame_full = 0 + int(np.linspace(0,F-10, self.total_num_modes)[self.mode])

                if start_frame_full< 0:
                    start_frame_full = self.mode

                list_full = []

                list_full = start_frame_full + np.asarray(
                    [int(int(skip_frames_full) * f) for f in range(self.params.num_frames)])
                
                # set all values greater than frame_count to frame_count
                list_full = np.minimum(list_full, frame_count-1)
                # list_full = list_full[:frame_count]

                full_clip = []

                for i, frame_idx in enumerate(list_full):
                    frame = torchvision.io.read_image(frame_list[frame_idx])
                    if i == 0:
                        _, self.ori_reso_h, self.ori_reso_w = frame.shape
                        self.min_size = min(self.ori_reso_h, self.ori_reso_w)
                    full_clip.append(self.augmentation(frame))

            return full_clip, torch.tensor(list_full)
        except:
            # traceback.print_exc()
            return None, None

    def build_clip(self, vid_path):
        frame_count = -1
        try:
            decord_vr = decord.VideoReader(vid_path)
            frame_count = len(decord_vr)
            frame_id_list = np.linspace(0, frame_count-1, self.num_frames, dtype=int)
            video_data = decord_vr.get_batch(frame_id_list)
            # video_data = video_data.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
            video_data = video_data.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)
            if self.model_name == 'vjepa':
                # Get video outputs after augmenting frame by frame.
                video_outputs = torch.zeros((self.num_frames, 3, 224, 224))
                for i in range(self.num_frames):
                    video_outputs[i] = self.augmentation(video_data[i])
            elif self.model_name == 'clip':
                video_outputs = self.augmentation(video_data)
                video_outputs = video_outputs.permute(1, 0, 2, 3)  # (T, C, H, W) -> (C, T, H, W)

            return video_outputs, torch.tensor(frame_id_list)
        except:
            traceback.print_exc()
            return None, None


def collate_fn(batch):
    f_clip, label, frame_list, vid_path = [], [], [], []
    for item in batch:
        if not (item[0] == None or item[1] == None or item[2] == None):
            f_clip.append(item[0]) 
            label.append(item[1])
            frame_list.append(item[2])
            vid_path.append(item[3])

    f_clip = torch.stack(f_clip, dim=0)
    label = torch.tensor(label)
    frame_list = torch.stack(frame_list, dim=0)
    
    return f_clip, label, frame_list, vid_path


if __name__ == '__main__':
    vid_dir = cfg.ucf101_path + '/Videos'
    videos = glob.glob(vid_dir + '/*/*')
    all_paths = open(os.path.join(cfg.ucf101_path, 'ucfTrainTestlist', f'testlist01.txt'),'r').read().splitlines()
    all_paths = [os.path.basename(path) for path in all_paths]
    videos = [vid for vid in videos if os.path.basename(vid) in all_paths]
    videos = videos[:10]
    train_dataset = baseline_val_dataloader(video_list=videos)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=4)
    print(f'Length of training dataset: {len(train_dataset)}')
    t = time.time()

    for i, (clip, label, frame_list, vid_path) in enumerate(train_dataloader):
        if i % 10 == 0:
            print()
            clip = clip.permute(0, 1, 3, 4, 2)
            print(f'Full_clip shape is {clip.shape}')
            print(f'Label is {label}', flush=True)
            print(time.time() - t)
            continue
