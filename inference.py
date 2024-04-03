import pandas as pd
import torch
from languagebind import LanguageBind, to_device, transform_dict, LanguageBindImageTokenizer


if __name__ == '__main__':
    device = 'cuda:0'
    device = torch.device(device)
    clip_type = {
        'video': 'LanguageBind_Video_FT',  # also LanguageBind_Video
        #   'audio': 'LanguageBind_Audio_FT',  # also LanguageBind_Audio
        #   'thermal': 'LanguageBind_Thermal',
        #   'image': 'LanguageBind_Image',
        #   'depth': 'LanguageBind_Depth',
    }

    model = LanguageBind(clip_type=clip_type, cache_dir='./cache_dir')
    model = model.to(device)
    model.eval()
    pretrained_ckpt = f'LanguageBind/LanguageBind_Image'
    tokenizer = LanguageBindImageTokenizer.from_pretrained(pretrained_ckpt, cache_dir='./cache_dir/tokenizer_cache_dir')
    modality_transform = {c: transform_dict[c](model.modality_config[c]) for c in clip_type.keys()}

    video = ['a-woman-with-long-hair_cstart-0.0_gs-7.5_pre-depth-zoe_cscale-1.0_grid-2_pad-1_model-realisticVisionV60B1_v60B1VAE.mp4', 'videos/k700/dancing ballet/pWNwRfbprU0_000152_000162.mp4']
    # language = ['combing hair', 'brushing hair']
    language = ['tap dancing', 'dancing ballet']
    # class_names = pd.read_excel('action_classes.xlsx')['less_actions']
    # class_names.dropna(inplace=True)
    # language = []
    # language = class_names.tolist()

    inputs = {
      'video': to_device(modality_transform['video'](video), device),
      'language': to_device(tokenizer(language, max_length=77, padding='max_length',
                            truncation=True, return_tensors='pt'), device)
    #   'audio': to_device(modality_transform['audio'](audio), device),
    #   'depth': to_device(modality_transform['depth'](depth), device),
    #   'thermal': to_device(modality_transform['thermal'](thermal), device),
    }

    with torch.no_grad():
        embeddings = model(inputs)

    # print(embeddings['language'])

    print("Video x Text: \n",
          torch.softmax(embeddings['video'] @ embeddings['language'].T, dim=-1).detach().cpu().numpy())
#     print("Image x Text: \n",
#           torch.softmax(embeddings['image'] @ embeddings['language'].T, dim=-1).detach().cpu().numpy())
#     print("Depth x Text: \n",
#           torch.softmax(embeddings['depth'] @ embeddings['language'].T, dim=-1).detach().cpu().numpy())
#     print("Audio x Text: \n",
#           torch.softmax(embeddings['audio'] @ embeddings['language'].T, dim=-1).detach().cpu().numpy())
#     print("Thermal x Text: \n",
#           torch.softmax(embeddings['thermal'] @ embeddings['language'].T, dim=-1).detach().cpu().numpy())

#     print("Video x Audio: \n",
#           torch.softmax(embeddings['video'] @ embeddings['audio'].T, dim=-1).detach().cpu().numpy())
#     print("Image x Depth: \n",
#           torch.softmax(embeddings['image'] @ embeddings['depth'].T, dim=-1).detach().cpu().numpy())
#     print("Image x Thermal: \n",
#           torch.softmax(embeddings['image'] @ embeddings['thermal'].T, dim=-1).detach().cpu().numpy())

