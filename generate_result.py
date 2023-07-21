import os
import cv2
import torch
from torchvision import transforms
from model import VideoSaliencyModel
import argparse
from utils import *
from os.path import join
import wandb
from loss import *

parser = argparse.ArgumentParser()
parser.add_argument('--file_weight',default="../vinet_rootgrouped_32_combined_datasets.pt", type=str)
parser.add_argument('--nhead',default=4, type=int)
parser.add_argument('--num_encoder_layers',default=3, type=int)
parser.add_argument('--transformer_in_channel',default=32, type=int)
parser.add_argument('--save_path',default='/ssd_scratch/cvit/bhavberi/dhf1k/results', type=str)
parser.add_argument('--start_idx',default=-1, type=int)
parser.add_argument('--num_parts',default=4, type=int)

parser.add_argument('--path_indata',default='/ssd_scratch/cvit/sarthak395/DHF1K/val', type=str)
parser.add_argument('--multi_frame',default=0, type=int)
parser.add_argument('--decoder_upsample',default=1, type=int)
parser.add_argument('--num_decoder_layers',default=-1, type=int)
parser.add_argument('--num_hier',default=3, type=int)
parser.add_argument('--clip_size',default=32, type=int)
parser.add_argument('--dataset', default='DHF1K', type=str)

parser.add_argument('--grouped_conv',default=True, type=bool)
parser.add_argument('--root_grouping', default=True, type=bool)
parser.add_argument('--depth_grouping', default=False, type=bool)
    
args = parser.parse_args()
print(args)

# WANDB
config = {
        "model_type": "VideoSaliency",
        "Backbone": "S3D",
        "clip_size": args.clip_size,
        "dataset": args.dataset,
        "gpu_id": 0,
        "grouping": args.grouped_conv,
        "root_grouping": args.root_grouping,
        "depth_grouping": args.depth_grouping,
        "wandb_run_name": "_rootgrouped"
    }

wandb.init(
            entity = 'bhavberi',   # wandb username
            project ='vinet0',   # wandb project name. New project will be created if given project is missing.
            name = 'whole_video_rootgrouped_dhf1k',
            config = config         # Config dict
            )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
def validate(args):
    path_indata = args.path_indata
    file_weight = args.file_weight

    len_temporal = args.clip_size

    model = VideoSaliencyModel(
        use_upsample=bool(args.decoder_upsample),
        num_hier=args.num_hier,
        num_clips=args.clip_size,
        grouped_conv=args.grouped_conv,
        root_grouping=args.root_grouping,
        depth=args.depth_grouping,
        efficientnet=args.efficientnet
    )

    model.load_state_dict(torch.load(file_weight))

    model = model.to(device)
    torch.backends.cudnn.benchmark = False
    model.eval()

    list_indata = [d for d in os.listdir(path_indata) if os.path.isdir(os.path.join(path_indata, d))]
    list_indata.sort()

    if args.start_idx!=-1:
        _len = (1.0/float(args.num_parts))*len(list_indata)
        list_indata = list_indata[int((args.start_idx-1)*_len): int(args.start_idx*_len)]

    frame_sim_loss = 0
    frame_cc_loss = 0
    frame_kldiv_loss = 0
    frame_cnt = 0

    avg_video_sim_loss = 0
    avg_video_cc_loss = 0
    avg_video_kldiv_loss = 0
    num_videos = 0
    for dname in list_indata:
        print ('processing ' + dname, flush=True)
        list_frames = [f for f in os.listdir(os.path.join(path_indata, dname, 'frames')) if os.path.isfile(os.path.join(path_indata, dname, 'frames', f))]
        list_frames.sort()
        os.makedirs(join(args.save_path, dname), exist_ok=True)

        video_sim_loss = 0
        video_cc_loss = 0
        video_kldiv_loss = 0
        num_frames = 0

        # process in a sliding window fashion
        if len(list_frames) >= 2*len_temporal-1:

            snippet = []
            for i in range(len(list_frames)):
                torch_img, img_size = torch_transform(os.path.join(path_indata, dname, 'frames', list_frames[i]))

                snippet.append(torch_img)
                
                if i >= len_temporal-1:
                    clip = torch.FloatTensor(torch.stack(snippet, dim=0)).unsqueeze(0)
                    clip = clip.permute((0,2,1,3,4))

                    sim_loss, cc_loss, kldiv_loss = process(model, clip, path_indata, dname, list_frames[i], args, img_size)

                    if np.isnan(sim_loss) or np.isnan(cc_loss):
	                    print("1", dname, list_frames[i])
                        # print("No saliency")
                    else:
                        frame_sim_loss += sim_loss
                        frame_kldiv_loss += kldiv_loss
                        frame_cc_loss += cc_loss
                        
                        frame_cnt += 1

                        video_sim_loss += sim_loss
                        video_kldiv_loss += kldiv_loss
                        video_cc_loss += cc_loss
                      
                        num_frames += 1
                                                

                    # process first (len_temporal-1) frames
                    if i < 2*len_temporal-2:
                        sim_loss, cc_loss, kldiv_loss = process(model, torch.flip(clip, [2]), path_indata, dname, list_frames[i-len_temporal+1], args, img_size)
                        if np.isnan(sim_loss) or np.isnan(cc_loss):
                            print("2", dname, list_frames[i])
                            # print("No saliency")
					    else:
						    frame_sim_loss += sim_loss
							frame_kldiv_loss += kldiv_loss
							frame_cc_loss += cc_loss
							
							frame_cnt += 1

							video_sim_loss += sim_loss
							video_kldiv_loss += kldiv_loss
							video_cc_loss += cc_loss
							
							num_frames += 1

                    del snippet[0]
        else:
            print (' more frames are needed')
        
        num_videos += 1
		avg_video_sim_loss += video_sim_loss / num_frames
		avg_video_kldiv_loss += video_kldiv_loss / num_frames
		avg_video_cc_loss += video_cc_loss / num_frames
		
    print("SIM:", frame_sim_loss/frame_cnt)
	print("CC:", frame_cc_loss/frame_cnt)
	print("KLDIV:", frame_kldiv_loss/frame_cnt)
        
    data_to_log = {
        'SIM': frame_sim_loss/frame_cnt,
        'CC': frame_cc_loss/frame_cnt,
        'KLDIV': frame_kldiv_loss/frame_cnt
    }
    wandb.log(data_to_log)

def torch_transform(path):
    img_transform = transforms.Compose([
            transforms.Resize((224, 384)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
    ])
    img = Image.open(path).convert('RGB')
    sz = img.size
    img = img_transform(img)
    return img, sz

def blur(img):
    k_size = 11
    bl = cv2.GaussianBlur(img,(k_size,k_size),0)
    return torch.FloatTensor(bl)

def process(model, clip, path_inpdata, dname, frame_no, args, img_size):
    with torch.no_grad():
        smap = model(clip.to(device)).cpu().data[0]
    
    smap = smap.numpy()
    

    _id = frame_no.split('.')[0].split('_')[-1]
    gt = cv2.imread(join(path_inpdata , dname, 'maps'.format(_id)), 0)
    gt = torch.FloatTensor(gt).unsqueeze(0)

    smap = cv2.resize(smap, (gt.shape[1], gt.shape[0]))
    smap = blur(smap)
    smap = smap.unsqueeze(0)

    sim_loss = similarity(smap, gt)
    cc_loss = cc(smap, gt)
    kldiv_loss = kldiv(smap, gt)

    img_save(smap, join(args.save_path, dname, frame_no), normalize=True)

    return sim_loss , cc_loss , kldiv_loss


    
    validate(args)

