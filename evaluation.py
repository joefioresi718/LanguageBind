import glob
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from languagebind import LanguageBind, to_device, transform_dict, LanguageBindImageTokenizer
import warnings
warnings.filterwarnings('ignore')


# Function to calculate metrics
def evaluate_performance(actual, predicted):
    average = 'binary'
    print(actual, predicted)
    accuracy = accuracy_score(actual, predicted)
    precision = precision_score(actual, predicted, average=average)
    recall = recall_score(actual, predicted, average=average)
    f1 = f1_score(actual, predicted, average=average)
    return accuracy, precision, recall, f1

# Function to calculate metrics
def evaluate_performance_class(actual, predicted, cls=None):
    average = None
    accuracy = accuracy_score(actual, predicted)
    precision = precision_score(actual, predicted, labels=cls, average=average)
    recall = recall_score(actual, predicted, labels=cls, average=average)
    f1 = f1_score(actual, predicted, labels=cls, average=average)
    return accuracy, precision, recall, f1


if __name__ == '__main__':
    device = 'cuda:0'
    device = torch.device(device)
    clip_type = {
        'video': 'LanguageBind_Video_FT',  # also LanguageBind_Video
    }

    model = LanguageBind(clip_type=clip_type, cache_dir='./cache_dir')
    model = model.to(device)
    model.eval()
    pretrained_ckpt = f'LanguageBind/LanguageBind_Image'
    tokenizer = LanguageBindImageTokenizer.from_pretrained(pretrained_ckpt, cache_dir='./cache_dir/tokenizer_cache_dir')
    modality_transform = {c: transform_dict[c](model.modality_config[c]) for c in clip_type.keys()}

    classes = [
        'javelin throw', 'baton twirling',
        'being excited', 'laughing',
        'casting fishing line', 'yarn spinning',
        'catching or throwing baseball', 'catching or throwing softball',
        'push', 'chew',
        'combing hair', 'brushing hair',
        'playing scrabble', 'doing sudoku',
        'high jump', 'hurdling',
        'skateboarding', 'longboarding',
        'making horseshoes', 'making jewelry',
        'playing cricket', 'playing polo',
        'pouring beer', 'pouring wine',
        'putting in contact lenses', 'putting on eyeliner',
        'surfing water', 'splashing water',
        'sucking lolly', 'licking',
        'tap dancing', 'dancing ballet',
        'tasting food', 'tasting wine',
        'washing hair', 'washing face',
    ]

    rel_idx = 20
    relevant_classes = classes[rel_idx:rel_idx+2]
    
    videos = []
    for cls in relevant_classes:
        videos.extend(glob.glob(f'videos/k700/{cls}/*'))

    inputs = {
      'video': to_device(modality_transform['video'](videos), device),
      'language': to_device(tokenizer(relevant_classes, max_length=77, padding='max_length',
                            truncation=True, return_tensors='pt'), device)
    }

    with torch.no_grad():
        embeddings = model(inputs)

    x_embeddings = torch.softmax(embeddings['video'] @ embeddings['language'].T, dim=-1)

    print("Video x Text: \n",
          x_embeddings.detach().cpu().numpy())
    
    class_idx = torch.argmax(x_embeddings, dim=-1)
    print(class_idx)

    results = {'pred': [], 'gt': [], 'filename': []}
    for i in range(len(class_idx)):
        print(f'{videos[i]}: {relevant_classes[class_idx[i]]}, {videos[i].split("/")[-2]}, {x_embeddings[i]}')
        results['pred'].append(class_idx[i].item())
        results['gt'].append(relevant_classes.index(videos[i].split("/")[-2]))
        results['filename'].append(videos[i].split("/")[-1])
    
    print('\nOverall Performance Metrics:')
    accuracy, precision, recall, f1 = evaluate_performance(results['gt'], results['pred'])
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    print(classification_report(results['gt'], results['pred'], target_names=relevant_classes))
    exit()
    
    relevant_class_idx = [relevant_classes.index(i) for i in relevant_classes]

    def custom_evaluate(predicted, label, other_label):
        n = len(predicted)
        print(predicted, label)
        tp = len(predicted[predicted == label])
        fp = len(predicted[predicted != label])
        fn = len(predicted[predicted == other_label])
        prec = tp / (tp + fp)
        print(n, tp, fp, prec, fn)
        return tp
    
    print('\nPerformance by Class:')
    for idx, i in enumerate(relevant_class_idx):
        class_results = {'pred': [], 'gt': [], 'filename': []}
        for j in range(len(results['pred'])):
            if results['gt'][j] == i:
                class_results['pred'].append(results['pred'][j])
                class_results['gt'].append(results['gt'][j])
                class_results['filename'].append(results['filename'][j])
        # results = custom_evaluate(class_results['pred'], i, relevant_class_idx[1-idx])
        accuracy, precision, recall, f1 = evaluate_performance(class_results['gt'], class_results['pred'])
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")



    # accuracy, precision, recall, f1 = evaluate_performance_class(class_results['gt'], class_results['pred'], relevant_classes)
    # print(f"{classes[i]}")
    # print(f"Accuracy: {accuracy:.4f}")
    # print(f"Precision: {precision:.4f}")
    # print(f"Recall: {recall:.4f}")
    # print(f"F1-Score: {f1:.4f}")
    # print(precision, recall, f1)
