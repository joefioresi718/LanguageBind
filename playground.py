import os.path
import torch
import torch.nn.functional as F
from languagebind import LanguageBind, to_device, transform_dict, LanguageBindImageTokenizer
from v_cls.zeroshot_cls import zero_shot_eval
from params import parse_args


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

def build_classifier(model, tokenizer, device):
    templates = (
        lambda c: f'a bad video of a {c}.',
        lambda c: f'a video of many {c}.',
        lambda c: f'a sculpture of a {c}.',
        lambda c: f'a video of the hard to see {c}.',
        lambda c: f'a low resolution video of the {c}.',
        lambda c: f'a rendering of a {c}.',
        lambda c: f'graffiti of a {c}.',
        lambda c: f'a bad video of the {c}.',
        lambda c: f'a cropped video of the {c}.',
        lambda c: f'a tattoo of a {c}.',
        lambda c: f'the embroidered {c}.',
        lambda c: f'a video of a hard to see {c}.',
        lambda c: f'a bright video of a {c}.',
        lambda c: f'a video of a clean {c}.',
        lambda c: f'a video of a dirty {c}.',
        lambda c: f'a dark video of the {c}.',
        lambda c: f'a drawing of a {c}.',
        lambda c: f'a video of my {c}.',
        lambda c: f'the plastic {c}.',
        lambda c: f'a video of the cool {c}.',
        lambda c: f'a close-up video of a {c}.',
        lambda c: f'a black and white video of the {c}.',
        lambda c: f'a painting of the {c}.',
        lambda c: f'a painting of a {c}.',
        lambda c: f'a pixelated video of the {c}.',
        lambda c: f'a sculpture of the {c}.',
        lambda c: f'a bright video of the {c}.',
        lambda c: f'a cropped video of a {c}.',
        lambda c: f'a plastic {c}.',
        lambda c: f'a video of the dirty {c}.',
        lambda c: f'a jpeg corrupted video of a {c}.',
        lambda c: f'a blurry video of the {c}.',
        lambda c: f'a video of the {c}.',
        lambda c: f'a good video of the {c}.',
        lambda c: f'a rendering of the {c}.',
        lambda c: f'a {c} in a video game.',
        lambda c: f'a video of one {c}.',
        lambda c: f'a doodle of a {c}.',
        lambda c: f'a close-up video of the {c}.',
        lambda c: f'a video of a {c}.',
        lambda c: f'the origami {c}.',
        lambda c: f'the {c} in a video game.',
        lambda c: f'a sketch of a {c}.',
        lambda c: f'a doodle of the {c}.',
        lambda c: f'a origami {c}.',
        lambda c: f'a low resolution video of a {c}.',
        lambda c: f'the toy {c}.',
        lambda c: f'a rendition of the {c}.',
        lambda c: f'a video of the clean {c}.',
        lambda c: f'a video of a large {c}.',
        lambda c: f'a rendition of a {c}.',
        lambda c: f'a video of a nice {c}.',
        lambda c: f'a video of a weird {c}.',
        lambda c: f'a blurry video of a {c}.',
        lambda c: f'a cartoon {c}.',
        lambda c: f'art of a {c}.',
        lambda c: f'a sketch of the {c}.',
        lambda c: f'a embroidered {c}.',
        lambda c: f'a pixelated video of a {c}.',
        lambda c: f'itap of the {c}.',
        lambda c: f'a jpeg corrupted video of the {c}.',
        lambda c: f'a good video of a {c}.',
        lambda c: f'a plushie {c}.',
        lambda c: f'a video of the nice {c}.',
        lambda c: f'a video of the small {c}.',
        lambda c: f'a video of the weird {c}.',
        lambda c: f'the cartoon {c}.',
        lambda c: f'art of the {c}.',
        lambda c: f'a drawing of the {c}.',
        lambda c: f'a video of the large {c}.',
        lambda c: f'a black and white video of a {c}.',
        lambda c: f'the plushie {c}.',
        lambda c: f'a dark video of a {c}.',
        lambda c: f'itap of a {c}.',
        lambda c: f'graffiti of the {c}.',
        lambda c: f'a toy {c}.',
        lambda c: f'itap of my {c}.',
        lambda c: f'a video of a cool {c}.',
        lambda c: f'a video of a small {c}.',
        lambda c: f'a tattoo of the {c}.',
    )

    classnames = [
        'ApplyEyeMakeup', 'ApplyLipstick', 'Archery', 'BabyCrawling', 'BalanceBeam', 
        'BandMarching', 'BaseballPitch', 'Basketball', 'BasketballDunk', 'BenchPress', 
        'Biking', 'Billiards', 'BlowDryHair', 'BlowingCandles', 'BodyWeightSquats', 
        'Bowling', 'BoxingPunchingBag', 'BoxingSpeedBag', 'BreastStroke', 'BrushingTeeth', 
        'CleanAndJerk', 'CliffDiving', 'CricketBowling', 'CricketShot', 'CuttingInKitchen', 
        'Diving', 'Drumming', 'Fencing', 'FieldHockeyPenalty', 'FloorGymnastics', 
        'FrisbeeCatch', 'FrontCrawl', 'GolfSwing', 'Haircut', 'Hammering', 
        'HammerThrow', 'HandstandPushups', 'HandstandWalking', 'HeadMassage', 'HighJump', 
        'HorseRace', 'HorseRiding', 'HulaHoop', 'IceDancing', 'JavelinThrow', 
        'JugglingBalls', 'JumpingJack', 'JumpRope', 'Kayaking', 'Knitting', 
        'LongJump', 'Lunges', 'MilitaryParade', 'Mixing', 'MoppingFloor', 
        'Nunchucks', 'ParallelBars', 'PizzaTossing', 'PlayingCello', 'PlayingDaf', 
        'PlayingDhol', 'PlayingFlute', 'PlayingGuitar', 'PlayingPiano', 'PlayingSitar', 
        'PlayingTabla', 'PlayingViolin', 'PoleVault', 'PommelHorse', 'PullUps', 
        'Punch', 'PushUps', 'Rafting', 'RockClimbingIndoor', 'RopeClimbing', 
        'Rowing', 'SalsaSpin', 'ShavingBeard', 'Shotput', 'SkateBoarding', 
        'Skiing', 'Skijet', 'SkyDiving', 'SoccerJuggling', 'SoccerPenalty', 
        'StillRings', 'SumoWrestling', 'Surfing', 'Swing', 'TableTennisShot', 
        'TaiChi', 'TennisSwing', 'ThrowDiscus', 'TrampolineJumping', 'Typing', 
        'UnevenBars', 'VolleyballSpiking', 'WalkingWithDog', 'WallPushups', 'WritingOnBoard', 
        'YoYo'
    ]

    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template(classname) for template in templates]
            texts = tokenizer(texts, max_length=77, padding='max_length',
                              truncation=True, return_tensors='pt').to(device)  # tokenize
            inputs = {'language': texts}
            class_embeddings = model(inputs)['language']
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)

    return zeroshot_weights


def get_data(args):
    data = {}
    data["v_cls"] = []

    from v_cls import get_video_cls_dataloader
    val_v_cls_data = 'UCF101'
    args.val_v_cls_data = val_v_cls_data
    args.video_data_path = '/home/jo869742/PythonProjects/datasets/UCF101'
    args.nb_classes = 101
    args.data_root = args.video_data_path + '/Videos'
    args.data_set = val_v_cls_data
    args.dist_eval = True
    args.sampling_rate = 8
    args.num_sample = 1
    args.test_num_segment = 5
    args.test_num_crop = 3
    args.num_workers = 4
    data['v_cls'].append({val_v_cls_data: get_video_cls_dataloader(args)})
    return data

if __name__ == '__main__':
    device = 'cuda:0'
    device = torch.device(device)
    clip_type = {
        'video': 'LanguageBind_Video',  # also LanguageBind_Video_FT
    }

    model = LanguageBind(clip_type=clip_type, cache_dir='./cache_dir')

    model = model.to(device)
    model.eval()
    pretrained_ckpt = f'LanguageBind/LanguageBind_Image'
    tokenizer = LanguageBindImageTokenizer.from_pretrained(pretrained_ckpt, cache_dir='./cache_dir/tokenizer_cache_dir')
    modality_transform = {c: transform_dict[c](model.modality_config[c]) for c in clip_type.keys()}

    classifier = build_classifier(model, tokenizer, device)

    args = parse_args()

    dataloader = get_data(args)['v_cls'][0]['UCF101']

    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        for batch in dataloader:
            videos = batch[0]
            target = torch.stack([torch.tensor(int(i)) for i in batch[1]], dim=0)
            ids = batch[2]
            chunk_nb = batch[3]
            split_nb = batch[4]
            videos = videos.to(device)
            target = target.to(device)
            inputs = {
                'video': videos,
                # 'language':
            }
            # inputs['language'] = to_device(tokenizer(language, max_length=77, padding='max_length',
            #                                         truncation=True, return_tensors='pt'), device)
            output = model(inputs)
            video_features = output['video'] if isinstance(output, dict) else output[0]
            logits = 100. * video_features @ classifier
            output = logits
            # for i in range(output.size(0)):
            #     string = "{} {} {} {} {}\n".format(
            #         ids[i], str(output.data[i].cpu().numpy().tolist()),
            #         str(int(target[i].cpu().numpy())),
            #         str(int(chunk_nb[i].cpu().numpy())),
            #         str(int(split_nb[i].cpu().numpy())))

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += videos.size(0)

    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100

    print(f"Top-1 accuracy: {top1:.2f}")
    print(f"Top-5 accuracy: {top5:.2f}")

