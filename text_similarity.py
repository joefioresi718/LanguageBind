import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
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

    video = ['assets/video/0.mp4', 'assets/video/1.mp4']
    m_words = ['man', 'mans', 'men', 'boy', 'boys', 'father', 'fathers', 'son', 'sons', 'he', 'his', 'him']
    f_words = ['woman', 'womans', 'women', 'girl', 'girls', 'lady', 'ladies', 'mother', 'mothers', 'daughter', 'daughters', 'she', 'her', 'hers']
    class_names = pd.read_excel('action_classes.xlsx')['less_actions']
    class_names.dropna(inplace=True)
    language = []
    language.extend(m_words)
    language.extend(f_words)
    language.extend(class_names.tolist())

    inputs = {
    #   'video': to_device(modality_transform['video'](video), device),
        'language': to_device(tokenizer(language, max_length=77, padding='max_length',
                              truncation=True, return_tensors='pt'), device)
    }

    with torch.no_grad():
        embeddings = model(inputs)

    lang_embeddings = embeddings['language'].cpu()
    m_vectors = lang_embeddings[:len(m_words)]
    f_vectors = lang_embeddings[len(m_words):len(m_words)+len(f_words)]
    class_vectors = lang_embeddings[len(m_words)+len(f_words):]

    # Compute class similarities.
    class_similarities = cosine_similarity(class_vectors)
    for i in range(len(class_similarities)):
        for j in range(len(class_similarities)):
            if i == j:
                class_similarities[i][j] = 0
            if class_similarities[i][j] == 1:
                class_similarities[i][j] = 0

    sim_dict = {}
    for i in range(len(class_similarities)):
        idx = np.argmax(class_similarities[i])
        sim_dict[class_names[i]] = (class_names[idx], idx)
        # print(f'{class_names[i]}: {class_names[np.argmax(class_similarities[i])]}')

    for idx, (k, v) in enumerate(sim_dict.items()):
        vm_sim = np.mean([1 - spatial.distance.cosine(m_vec, class_vectors[v[1]]) for m_vec in m_vectors])
        vf_sim = np.mean([1 - spatial.distance.cosine(f_vec, class_vectors[v[1]]) for f_vec in f_vectors])
        vsim = vm_sim - vf_sim
        km_sim = np.mean([1 - spatial.distance.cosine(m_vec, class_vectors[idx]) for m_vec in m_vectors])
        kf_sim = np.mean([1 - spatial.distance.cosine(f_vec, class_vectors[idx]) for f_vec in f_vectors])
        ksim = km_sim - kf_sim
        if ksim * vsim < 0:
            print(f'{k}: {v[0]}, ksim: {ksim}, vsim: {vsim}')
    