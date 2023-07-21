import os
import cv2
import torch
from model import VideoSaliencyModel
import argparse

from utils import *
import time
from tqdm import tqdm
from torchvision import transforms
from os.path import join

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def validate(args):
    path_indata = args.path_indata
    file_weight = args.file_weight

    len_temporal = args.clip_size

    net_load_time = 0
    net_save_time = 0
    net_tensorify_time = 0
    net_process_time = 0
    net_enc_time = 0
    net_dec_time = 0

    load_calls = 0
    save_calls = 0
    tensor_calls = 0
    process_calls = 0

    if args.img_backbone in ['squeezenet', 's3d']:
        # model = VideoSaliencyModel(
        #     img_backbone=args.img_backbone,
        #     # transformer_in_channel=args.transformer_in_channel,
        #     # nhead=args.nhead,
        #     use_upsample=bool(args.decoder_upsample),
        #     num_hier=args.num_hier,
        #     num_clips=args.clip_size
        # )
        model = VideoSaliencyModel(
        use_upsample=bool(args.decoder_upsample),
        num_hier=args.num_hier,
        num_clips=args.clip_size,
        grouped_conv=args.grouped_conv,
        root_grouping=args.root_grouping,
        depth=args.depth_grouping,
        efficientnet=args.efficientnet
    )
    # elif 'tsm' in args.img_backbone:
    #     from tsm.ops.models import TSN
    #     model = TSN(base_model=args.img_backbone.split('_')[
    #                 1], shift_div=args.shift_div, is_shift=args.shift, shift_place=args.shift_place)
    # elif args.img_backbone == 'acar':
    #     from acar_modified_model import AVA_model
    #     model = AVA_model()

    # if args.img_backbone != 'acar':
    #     model.load_state_dict(torch.load(file_weight))
    # else:
    model = torch.nn.DataParallel(model)
    # model.load_state_dict(torch.load(file_weight)['state_dict'])
    model.load_state_dict(torch.load(file_weight)['state_dict'])


    model = model.to(device).half()
    torch.backends.cudnn.benchmark = True
    model.eval()

    list_indata = [d for d in os.listdir(
        path_indata) if os.path.isdir(os.path.join(path_indata, d))]
    list_indata.sort()
    print(list_indata)

    if args.start_idx != -1:
        _len = (1.0/float(args.num_parts))*len(list_indata)
        list_indata = list_indata[int(
            (args.start_idx-1)*_len): int(args.start_idx*_len)]

    pbar = tqdm(len(list_indata))
    for dname in list_indata:
        list_frames = [f for f in os.listdir(os.path.join(path_indata, dname, 'frames')) if os.path.isfile(
            os.path.join(path_indata, dname, 'frames', f))]
        list_frames.sort()
        os.makedirs(join(args.save_path, dname), exist_ok=True)

        pbar.set_description(
            f'processing {dname} ({len(list_frames)}). Load times: {net_load_time/(load_calls+1e-6):.3f}.  Save times: {net_save_time/(save_calls+1e-6):.3f}.  Tensorify times: {net_tensorify_time/(tensor_calls+1e-6):.3f}. Process time: {net_process_time/(process_calls+1e-6):.3f} ({net_enc_time/(process_calls+1e-6):.3f}+{net_dec_time/(process_calls+1e-6):.3f})')
        pbar.refresh()

        # process in a sliding window fashion
        if len(list_frames) >= 2*len_temporal-1:

            snippet = []
            pbar2 = tqdm(total=len(list_frames))
            pbar.set_description(f'{dname}: ')

            output_tensor = torch.empty((0,), dtype=torch.half, device='cuda')
            paths = list()
            for i in range(len(list_frames)):

                curr_load_time = time.time()
                torch_img, img_size = torch_transform(os.path.join(
                    path_indata, dname, 'frames', list_frames[i]))

                snippet.append(torch_img)
                curr_load_time = time.time()-curr_load_time

                net_load_time += curr_load_time
                load_calls += 1

                if i >= len_temporal-1:
                    curr_tensorify_time = time.time()
                    clip = torch.tensor(
                        torch.stack(snippet, dim=0), dtype=torch.half, device=device).unsqueeze(0)
                    clip = clip.permute((0, 2, 1, 3, 4))
                    curr_tensorify_time = time.time() - curr_tensorify_time
                    net_tensorify_time += curr_tensorify_time
                    tensor_calls += 1

                    curr_process_time = time.time()
                    output, times = process(model, clip, path_indata, dname,
                                            list_frames[i], args, img_size)
                    output_tensor = torch.cat((output_tensor, output), dim=0)
                    paths.append(join(args.save_path, dname, list_frames[i]))
                    curr_process_time = time.time()-curr_process_time
                    net_process_time += curr_process_time
                    net_enc_time += times[0]
                    net_dec_time += times[1]
                    process_calls += 1

                    curr_save_time = time.time()
                    save(output, path_indata, dname,
                         list_frames[i], args, img_size)
                    curr_save_time = time.time()-curr_save_time

                    net_save_time += curr_save_time
                    save_calls += 1

                    pbar2.update(1)

                    # process first (len_temporal-1) frames
                    if i < 2*len_temporal-2:
                        curr_process_time = time.time()
                        output, times = process(model, torch.flip(
                            clip, [2]), path_indata, dname, list_frames[i-len_temporal+1], args, img_size)
                        output_tensor = torch.cat(
                            (output_tensor, output), dim=0)
                        paths.append(join(args.save_path, dname,
                                     list_frames[i-len_temporal+1]))
                        curr_process_time = time.time()-curr_process_time
                        net_process_time += curr_process_time
                        net_enc_time += times[0]
                        net_dec_time += times[1]
                        process_calls += 1

                        curr_save_time = time.time()
                        save(output, path_indata, dname,
                             list_frames[i-len_temporal+1], args, img_size)
                        curr_save_time = time.time()-curr_save_time

                        net_save_time += curr_save_time
                        save_calls += 1

                        pbar2.update(1)

                    del snippet[0]

                if output_tensor.size(0) >= args.save_every:
                    bulk_save(output_tensor, paths, img_size)

                    output_tensor = torch.empty(
                        (0,), dtype=torch.half, device=device)
                    paths = list()

                pbar.set_description(
                    f'processing {dname} ({len(list_frames)}). Load times: {net_load_time/(load_calls+1e-6):.3f}.  Save times: {net_save_time/(save_calls+1e-6):.3f}. Tensorify times: {net_tensorify_time/(tensor_calls+1e-6):.3f}. Process time: {net_process_time/(process_calls+1e-6):.3f} ({net_enc_time/(process_calls+1e-6):.3f}+{net_dec_time/(process_calls+1e-6):.3f})')
                pbar.refresh()

            if output_tensor.size(0) >= 0:
                bulk_save(output_tensor, paths, img_size)

                output_tensor = torch.empty(
                    (0,), dtype=torch.half, device=device)
                paths = list()

        else:
            print(f'{dname}: more frames are needed')
        pbar.update(1)

    pbar.set_description(
        f'Done. Load times: {net_load_time/(load_calls+1e-6):.3f}.  Save times: {net_save_time/(save_calls+1e-6):.3f}.  Tensorify times: {net_tensorify_time/(tensor_calls+1e-6):.3f}. Process time: {net_process_time/(process_calls+1e-6):.3f} ({net_enc_time/(process_calls+1e-6):.3f}+{net_dec_time/(process_calls+1e-6):.3f})')
    pbar.refresh()


class Scale(object):
    """Rescale the input PIL.Image to the given size.
    Args:
        resize (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size, size * height / width).
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``.
        max_ratio (float, optional): If not None, denotes maximum allowed aspect
            ratio after rescaling the input.
    """

    def __init__(self, resize, interpolation=Image.BILINEAR, max_ratio=None):
        assert isinstance(resize,
                          int) or (isinstance(resize, collections.Iterable) and
                                   len(resize) == 2)
        self.resize = resize
        self.interpolation = interpolation
        self.max_ratio = max_ratio

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be scaled.
        Returns:
            PIL.Image: Rescaled image.
        """
        w, h = img.size
        if isinstance(self.size, int):
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return img.resize((ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return img.resize((ow, oh), self.interpolation)
        else:
            if w == self.size[0] and h == self.size[1]:
                return img
            return img.resize(self.size, self.interpolation)

    def randomize_parameters(self, size=None):
        if isinstance(self.resize, int) and size and self.max_ratio:
            ratio = max(size[0] / size[1], size[1] / size[0])
            if ratio > self.max_ratio:
                if size[0] > size[1]:
                    resize = (int(self.resize * self.max_ratio), self.resize)
                else:
                    resize = (self.resize, int(self.resize * self.max_ratio))
            else:
                resize = self.resize
        else:
            resize = self.resize
        self.size = resize
        return [{'transform': 'Scale', 'size': self.size}]

    def __repr__(self):
        return '{self.__class__.__name__}(resize={self.resize}, interpolation={self.interpolation}, max_ratio={self.max_ratio})'.format(self=self)


def torch_transform(path):
    scale = Scale(resize=256)
    params = scale.randomize_parameters()
    img_transform = transforms.Compose([
        scale,
        transforms.ToTensor(),
        transforms.Normalize(
            [0.450, 0.450, 0.450],
            [0.225, 0.225, 0.225]
        )
    ])
    # img_transform = transforms.Compose([
    #     transforms.Resize((224, 384)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(
    #         [0.485, 0.456, 0.406],
    #         [0.229, 0.224, 0.225]
    #     )
    # ])
    img = Image.open(path).convert('RGB')
    sz = img.size
    img = img_transform(img)
    return img, sz


def blur(img):
    k_size = 11
    bl = cv2.GaussianBlur(img, (k_size, k_size), 0)
    return torch.FloatTensor(bl)


def process_old(model, clip, path_inpdata, dname, frame_no, args, img_size):
    with torch.no_grad():
        smap = model(clip).cpu()

    return smap


def process(model, clip, path_inpdata, dname, frame_no, args, img_size):
    with torch.no_grad():
        smap = model(clip)
        # smap, times = model.benchmark_forward(clip)
        smap = smap

    return smap, [0, 0]


def save(smap, path_inpdata, dname, frame_no, args, img_size):
    return
    smap = smap.cpu().numpy()[0]
    smap = cv2.resize(smap, (img_size[0], img_size[1]))
    smap = blur(smap)

    img_save(smap, join(args.save_path, dname, frame_no), normalize=True)


def bulk_save(output_tensor, paths, img_size):

    # Convert the output tensor to a numpy array on the CPU
    output_array = output_tensor.cpu().float().numpy()

    # Loop over the frames in the output tensor
    for i in range(output_tensor.size(0)):

        # Get the i-th frame from the output tensor
        frame = output_array[i]

        # Resize and blur the frame
        frame = cv2.resize(frame, (img_size[0], img_size[1]))
        frame = blur(frame)

        # Generate a filename for the output frame
        filepath = paths[i]

        # Save the output frame as an image
        img_save(frame, filepath, normalize=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--file_weight', default="./saved_models/ViNet_DHF1K.pt", type=str)
    parser.add_argument('--save_every', default=600, type=int)
    parser.add_argument('--img_backbone', default='s3d',
                        choices=['s3d', 'squeezenet', 'tsm_resnet50', 'tsm_mobilenetv2', 'acar'], type=str)
    parser.add_argument('--nhead', default=4, type=int)
    parser.add_argument('--num_encoder_layers', default=3, type=int)
    parser.add_argument('--transformer_in_channel', default=32, type=int)
    parser.add_argument(
        '--save_path', default='/ssd_scratch/cvit/sarthak395/results/theatre_hollywood', type=str)
    parser.add_argument('--start_idx', default=-1, type=int)
    parser.add_argument('--num_parts', default=4, type=int)
    parser.add_argument(
        '--path_indata', default='/ssd_scratch/cvit/sarthak395/DHF1K/val', type=str)
    parser.add_argument('--multi_frame', default=0, type=int)
    parser.add_argument('--decoder_upsample', default=1, type=int)
    parser.add_argument('--num_decoder_layers', default=-1, type=int)
    parser.add_argument('--num_hier', default=3, type=int)
    parser.add_argument('--clip_size', default=32, type=int)

    parser.add_argument('--grouped_conv',default=False, type=bool)
    parser.add_argument('--root_grouping', default=False, type=bool)
    parser.add_argument('--depth_grouping', default=False, type=bool)

    parser.add_argument('--shift', default=False,
                        action="store_true", help='use shift for models')
    parser.add_argument('--shift_div', default=8, type=int,
                        help='number of div for shift (default: 8)')
    parser.add_argument('--shift_place', default='blockres',
                        type=str, help='place for shift (default: stageres)')

    args = parser.parse_args()
    print(args)
    validate(args)
