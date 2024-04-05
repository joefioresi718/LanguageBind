import numpy as np
import os.path
import pandas as pd
import torch
import decord
# from torchvision.transforms import v2
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
from languagebind import LanguageBind, to_device, transform_dict, LanguageBindImageTokenizer, LanguageBindVideoTokenizer
import glob

import config as cfg
from load_vjepa import init_vjepa
from dataloader import baseline_val_dataloader, collate_fn

decord.bridge.set_bridge('torch')


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def clip_similarities(video_list, dataset_name):
    device = 'cuda:0'
    device = torch.device(device)
    # CLIP model similarities.
    clip_type = {
        'video': 'LanguageBind_Video_FT',  # also LanguageBind_Video
    }

    c_model = LanguageBind(clip_type=clip_type, cache_dir='./cache_dir')
    c_model = c_model.to(device)
    c_model.eval()
    # pretrained_ckpt = f'LanguageBind/LanguageBind_Video'
    # # tokenizer = LanguageBindImageTokenizer.from_pretrained(pretrained_ckpt, cache_dir='./cache_dir/tokenizer_cache_dir')
    # tokenizer = LanguageBindVideoTokenizer.from_pretrained(pretrained_ckpt, cache_dir='./cache_dir/tokenizer_cache_dir')
    # modality_transform = {c: transform_dict[c](c_model.modality_config[c]) for c in clip_type.keys()}

    # c_inputs = {
    #   'video': to_device(modality_transform['video'](video_list), device),
    #     # 'language': to_device(tokenizer(language, max_length=77, padding='max_length',
    #     #                       truncation=True, return_tensors='pt'), device)
    # }

    # with torch.inference_mode():
    #     c_embeddings = c_model(c_inputs)

    # c_vid_embeddings = c_embeddings['video'].detach().cpu()

    batch_size = 128
    # Custom dataset.
    c_dataset = baseline_val_dataloader(video_list, dataset=dataset_name, model_name='clip', num_frames=8, shuffle=False)
    c_loader = DataLoader(c_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=8)
    print(f'Dataset length: {len(c_dataset)}, Number of batches: {len(c_loader)}')

    c_vid_embeddings = torch.zeros((len(c_dataset), 1024))
    vid_paths = []
    bad_idxs = []
    with torch.inference_mode():
        for i, (clips, _, _, vid_path) in enumerate(c_loader):
            vid_paths.append(vid_path)
            clips = clips.cuda()
            bs = clips.shape[0]
            clips = {'video': {'pixel_values': clips}}
            # TODO: this will pseudo-break if any single video fails.
            c_vid_embeddings[i*batch_size:i*batch_size+bs] = c_model(clips)['video'].detach().cpu()
            if bs < batch_size:
                bad_idxs.extend(list(range(i*batch_size+bs, i*batch_size+batch_size)))

    bad_idxs = torch.tensor(bad_idxs)
    mask = torch.ones(c_vid_embeddings.size(), dtype=torch.bool)
    mask[bad_idxs] = False

    # Apply mask
    c_vid_embeddings = c_vid_embeddings[mask]

    # c_vid_embeddings = torch.tensor([v for i, v in enumerate(c_vid_embeddings) if i not in bad_idxs])

    del c_model
    return c_vid_embeddings, vid_paths


def video_similarities(video_list, dataset_name):
    # Video model similarity.
    encoder, classifier = init_vjepa()
    classifier.linear = torch.nn.Identity()
    encoder.eval(), classifier.eval()
    num_features = encoder.embed_dim
    v_model = torch.nn.Sequential(encoder, classifier)

    batch_size = 16
    # Custom dataset.
    v_dataset = baseline_val_dataloader(video_list, dataset=dataset_name, model_name='vjepa', num_frames=8, shuffle=False)
    v_loader = DataLoader(v_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=8)
    print(f'Dataset length: {len(v_dataset)}, Number of batches: {len(v_loader)}')

    v_vid_embeddings = torch.zeros((len(v_dataset), num_features))
    bad_idxs = []
    with torch.inference_mode():
        for i, (clips, _, _, _) in enumerate(v_loader):
            clips = clips.cuda().permute(0, 2, 1, 3, 4)
            bs = clips.shape[0]
            v_vid_embeddings[i*batch_size:i*batch_size+bs] = v_model(clips).detach().cpu()
            if bs < batch_size:
                bad_idxs.extend(list(range(i*batch_size+bs, i*batch_size+batch_size)))

    bad_idxs = torch.tensor(bad_idxs)
    mask = torch.ones(v_vid_embeddings.size(), dtype=torch.bool)
    mask[bad_idxs] = False

    v_vid_embeddings = v_vid_embeddings[mask]

    # v_vid_embeddings = torch.tensor([v for i, v in enumerate(v_vid_embeddings) if i not in bad_idxs])

    del encoder, classifier, v_model 
    return v_vid_embeddings


if __name__ == '__main__':
    # videos = ['assets/video/0.mp4', 'assets/video/1.mp4']
    # vid_dir = cfg.ucf101_path + '/Videos'
    # video_list = glob.glob(vid_dir + '/*/*')
    # all_paths = open(os.path.join(cfg.ucf101_path, 'ucfTrainTestlist', f'testlist01.txt'),'r').read().splitlines()
    # all_paths = [os.path.basename(path) for path in all_paths]
    # video_list = sorted([vid for vid in video_list if os.path.basename(vid) in all_paths])
    # video_list = video_list[:100]

    video_list = None
    dataset_name = 'k400'

    c_vid_embeddings, video_list = clip_similarities(video_list, dataset_name)
    v_vid_embeddings = video_similarities(video_list, dataset_name)

    # if video_list is None:
    #     dl = baseline_val_dataloader(None, dataset=dataset_name, shuffle=False)
    #     video_list = dl.data

    # Compute class similarities.
    c_vid_similarities = cosine_similarity(c_vid_embeddings)
    c_vid_similarities = np.triu(c_vid_similarities, k=1)
    v_vid_similarities = cosine_similarity(v_vid_embeddings)
    v_vid_similarities = np.triu(v_vid_similarities, k=1)

    v1, v2, c_score, v_score = [], [], [], []

    # Print all high similarities.
    for i, vid1 in enumerate(video_list):
        for j, vid2 in enumerate(video_list):
            vid1 = os.path.basename(vid1)
            vid2 = os.path.basename(vid2)
            if i != j and c_vid_similarities[i][j] > 0.90 and v_vid_similarities[i][j] < 0.6:
                print(f'{vid1} and {vid2} have high CLIP similarity {c_vid_similarities[i][j]} and low video similarity {v_vid_similarities[i][j]}')
                v1.append(vid1)
                v2.append(vid2)
                c_score.append(f'{c_vid_similarities[i][j]:.05f}')
                v_score.append(f'{v_vid_similarities[i][j]:.05f}')
            # else:
            #     if c_vid_similarities[i][j] == 0.0:
            #         continue
            #     print(f'{vid1} and {vid2} have low similarity {c_vid_similarities[i][j]}')
                
    import pandas as pd

    df = pd.DataFrame({'video1': v1, 'video2': v2, 'clip_similarity': c_score, 'vssl_similarity': v_score})
    df.to_csv('CLIPblind_pairs.csv', index=False)
    