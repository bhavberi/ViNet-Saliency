import os
import torch.utils.data as data
import time
import torch
import numpy as np
from PIL import Image
import cv2
import torch.nn.functional as F

import json

def get_aug_info(init_size, params):
    size = init_size
    bbox = [0.0, 0.0, 1.0, 1.0]
    flip = False

    for t in params:
        if t is None:
            continue

        if t['transform'] == 'RandomHorizontalFlip':
            if t['flip']:
                flip = not flip
            continue

        if t['transform'] == 'Scale':
            if isinstance(t['size'], int):
                w, h = size
                if (w <= h and w == t['size']) or (h <= w and h == t['size']):
                    continue
                if w < h:
                    ow = t['size']
                    oh = int(t['size'] * h / w)
                    size = [ow, oh]
                else:
                    oh = t['size']
                    ow = int(t['size'] * w / h)
                    size = [ow, oh]
            else:
                size = t['size']
            continue

        if t['transform'] == 'CenterCrop':
            w, h = size
            size = t['size']

            x1 = int(round((w - size[0]) / 2.))
            y1 = int(round((h - size[1]) / 2.))
            x2 = x1 + size[0]
            y2 = y1 + size[1]

        elif t['transform'] == 'CornerCrop':
            w, h = size
            size = [t['size']] * 2

            if t['crop_position'] == 'c':
                th, tw = (t['size'], t['size'])
                x1 = int(round((w - tw) / 2.))
                y1 = int(round((h - th) / 2.))
                x2 = x1 + tw
                y2 = y1 + th
            elif t['crop_position'] == 'tl':
                x1 = 0
                y1 = 0
                x2 = t['size']
                y2 = t['size']
            elif t['crop_position'] == 'tr':
                x1 = w - self.size
                y1 = 0
                x2 = w
                y2 = t['size']
            elif t['crop_position'] == 'bl':
                x1 = 0
                y1 = h - t['size']
                x2 = t['size']
                y2 = h
            elif t['crop_position'] == 'br':
                x1 = w - t['size']
                y1 = h - t['size']
                x2 = w
                y2 = h

        elif t['transform'] == 'ScaleJitteringRandomCrop':
            min_length = min(size[0], size[1])
            jitter_rate = float(t['scale']) / min_length

            w = int(jitter_rate * size[0])
            h = int(jitter_rate * size[1])
            size = [t['size']] * 2

            x1 = t['pos_x'] * (w - t['size'])
            y1 = t['pos_y'] * (h - t['size'])
            x2 = x1 + t['size']
            y2 = y1 + t['size']

        dl = float(x1) / w * (bbox[2] - bbox[0])
        dt = float(y1) / h * (bbox[3] - bbox[1])
        dr = float(x2) / w * (bbox[2] - bbox[0])
        db = float(y2) / h * (bbox[3] - bbox[1])

        if flip:
            bbox = [bbox[2] - dr, bbox[1] + dt, bbox[2] - dl, bbox[1] + db]
        else:
            bbox = [bbox[0] + dl, bbox[1] + dt, bbox[0] + dr, bbox[1] + db]

    return {'init_size': init_size, 'crop_box': bbox, 'flip': flip}


def batch_pad(images, alignment=1, pad_value=0):
    max_img_h = max([_.size(-2) for _ in images])
    max_img_w = max([_.size(-1) for _ in images])
    target_h = int(np.ceil(max_img_h / alignment) * alignment)
    target_w = int(np.ceil(max_img_w / alignment) * alignment)
    padded_images, pad_ratios = [], []
    for image in images:
        src_h, src_w = image.size()[-2:]
        pad_size = (0, target_w - src_w, 0, target_h - src_h)
        padded_images.append(
            F.pad(image, pad_size, 'constant', pad_value).data)
        pad_ratios.append([target_w / src_w, target_h / src_h])
    return torch.stack(padded_images), pad_ratios

class AVAVideoDataset(data.Dataset):
    def __init__(self, video_root, ann_file, remove_clips_without_annotations, frame_span, box_file=None,
                 eval_file_paths={}, box_thresh=0.0, action_thresh=0.0, transforms=None):

        print('loading annotations into memory...')
        tic = time.time()
        json_dict = json.load(open(ann_file, 'r'))
        assert type(json_dict) == dict, 'annotation file format {} not supported'.format(
            type(json_dict))
        print('Done (t={:0.2f}s)'.format(time.time() - tic))

        self.video_root = video_root
        self.transforms = transforms
        self.frame_span = frame_span

        # These two attributes are used during ava evaluation...
        # Maybe there is a better implementation
        self.eval_file_paths = eval_file_paths
        self.action_thresh = action_thresh

        clip2ann = defaultdict(list)
        if "annotations" in json_dict:
            for ann in json_dict["annotations"]:
                action_ids = ann["action_ids"]
                one_hot = np.zeros(81, dtype=np.bool)
                one_hot[action_ids] = True
                packed_act = np.packbits(one_hot[1:])
                clip2ann[ann["image_id"]].append(
                    dict(bbox=ann["bbox"], packed_act=packed_act))

        movies_size = {}
        clips_info = {}
        for img in json_dict["images"]:
            mov = img["movie"]
            if mov not in movies_size:
                movies_size[mov] = [img["width"], img["height"]]
            clips_info[img["id"]] = [mov, img["timestamp"]]
        self.movie_info = NpInfoDict(movies_size, value_type=np.int32)
        clip_ids = sorted(list(clips_info.keys()))

        if remove_clips_without_annotations:
            clip_ids = [clip_id for clip_id in clip_ids if clip_id in clip2ann]

        if box_file:
            # this is only for validation or testing
            # we use detected boxes, so remove clips without boxes detected.
            imgToBoxes = self.load_box_file(box_file, box_thresh)
            clip_ids = [
                img_id
                for img_id in clip_ids
                if len(imgToBoxes[img_id]) > 0
            ]
            self.det_persons = NpBoxDict(imgToBoxes, clip_ids,
                                         value_types=[("bbox", np.float32), ("score", np.float32)])
        else:
            self.det_persons = None

        self.anns = NpBoxDict(clip2ann, clip_ids, value_types=[
                              ("bbox", np.float32), ("packed_act", np.uint8)])

        clips_info = {  # key
            clip_id:
                [
                    self.movie_info.convert_key(clips_info[clip_id][0]),
                    clips_info[clip_id][1]
                ] for clip_id in clip_ids
        }
        self.clips_info = NpInfoDict(clips_info, value_type=np.int32)

    def __getitem__(self, idx):

        _, clip_info = self.clips_info[idx]

        # mov_id is the id in self.movie_info
        mov_id, timestamp = clip_info
        # movie_id is the human-readable youtube id.
        movie_id, movie_size = self.movie_info[mov_id]
        video_data = self._decode_video_data(movie_id, timestamp)

        im_w, im_h = movie_size

        if self.det_persons is None:
            # Note: During training, we only use gt. Thus we should not provide box file,
            # otherwise we will use only box file instead.

            boxes, packed_act = self.anns[idx]

            boxes_tensor = torch.as_tensor(
                boxes, dtype=torch.float32).reshape(-1, 4)  # guard against no boxes
            boxes = BoxList(boxes_tensor, (im_w, im_h),
                            mode="xywh").convert("xyxy") # top-left and bottom-right

            # Decode the packed bits from uint8 to one hot, since AVA has 80 classes,
            # it can be exactly denoted with 10 bytes, otherwise we may need to discard some bits.
            one_hot_label = np.unpackbits(packed_act, axis=1)
            one_hot_label = torch.as_tensor(one_hot_label, dtype=torch.uint8)

            boxes.add_field("labels", one_hot_label)  # 80

        else:
            boxes, box_score = self.det_persons[idx]
            boxes_tensor = torch.as_tensor(boxes).reshape(-1, 4)
            boxes = BoxList(boxes_tensor, (im_w, im_h),
                            mode="xywh").convert("xyxy")

            box_score_tensor = torch.as_tensor(
                box_score, dtype=torch.float32).reshape(-1, 1)
            boxes.add_field("scores", box_score_tensor)

        boxes = boxes.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            video_data, boxes, transform_randoms = self.transforms(
                video_data, boxes)

        return video_data, boxes, idx

    def get_video_info(self, index):
        _, clip_info = self.clips_info[index]
        # mov_id is the id in self.movie_info
        mov_id, timestamp = clip_info
        # movie_id is the human-readable youtube id.
        movie_id, movie_size = self.movie_info[mov_id]
        w, h = movie_size
        return dict(width=w, height=h, movie=movie_id, timestamp=timestamp)

    def load_box_file(self, box_file, score_thresh=0.0):
        import json

        print('Loading box file into memory...')
        tic = time.time()
        with open(box_file, "r") as f:
            box_results = json.load(f)
        print('Done (t={:0.2f}s)'.format(time.time() - tic))

        boxImgIds = [box['image_id'] for box in box_results]

        imgToBoxes = defaultdict(list)
        for img_id, box in zip(boxImgIds, box_results):
            if box['score'] >= score_thresh:
                imgToBoxes[img_id].append(box)
        return imgToBoxes

    def _decode_video_data(self, dirname, timestamp):
        # decode target video data from segment per second.

        video_folder = os.path.join(self.video_root, dirname)
        right_span = self.frame_span//2
        left_span = self.frame_span - right_span

        # load right
        cur_t = timestamp
        right_frames = []
        while len(right_frames) < right_span:
            video_path = os.path.join(video_folder, "{}.mp4".format(cur_t))
            # frames = cv2_decode_video(video_path)
            frames = av_decode_video(video_path)
            if len(frames) == 0:
                raise RuntimeError(
                    "Video {} cannot be decoded.".format(video_path))
            right_frames = right_frames+frames
            cur_t += 1

        # load left
        cur_t = timestamp-1
        left_frames = []
        while len(left_frames) < left_span:
            video_path = os.path.join(video_folder, "{}.mp4".format(cur_t))
            # frames = cv2_decode_video(video_path)
            frames = av_decode_video(video_path)
            if len(frames) == 0:
                raise RuntimeError(
                    "Video {} cannot be decoded.".format(video_path))
            left_frames = frames+left_frames
            cur_t -= 1

        # adjust key frame to center, usually no need
        min_frame_num = min(len(left_frames), len(right_frames))
        frames = left_frames[-min_frame_num:] + right_frames[:min_frame_num]

        video_data = np.stack(frames)

        return video_data

    def __len__(self):
        return len(self.clips_info)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Video Root Location: {}\n'.format(self.video_root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(
            tmp, self.transforms.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class MVVA(data.Dataset):
    def __init__(self,
                 gt_sal_maps_path,
                 bbox_root_path,
                 videos_frames_root_path,
                 spatial_transform=None,
                 temporal_transform=None,
                 mode='train',
                 clip_size=32,
                 alternate=1,
                 split = 1,
                 fold_lists_path = None):

        
        self.videos_frames_root_path = videos_frames_root_path
        self.gt_sal_maps_path = gt_sal_maps_path
        self.bbox_root_path = bbox_root_path

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform

    # All the video files are in the same folder
        self.video_files = sorted(os.listdir(self.videos_frames_root_path))

    # All the video names are the same as the video files without the extension
        self.video_names = [video_file.split('.')[0] for video_file in self.video_files]

        self.mode = mode

        if (self.mode == 'val' or self.mode == 'test'):
            mode = 'test'
        file_name = '{}_list_{}_{}_fps.txt'.format('mvva', mode, split)

        self.list_indata = []
        with open(os.path.join(fold_lists_path, file_name), 'r') as f:
            for line in f.readlines():
                name = line.split(' ')[0].strip()
				#num_frames = line.split(' ')[1].strip()
                self.list_indata.append(name)
		#list_indata = [i for i in list_indata if i in ['001']]
		#print(list_indata)
        self.video_names = [i for i in self.video_names if i in self.list_indata]

        self.video_files = [i for i in self.video_files if i.split('.')[0] in self.video_names]
		

        self.len_snippet = clip_size
        self.alternate = alternate

        if self.mode == 'val':

            self.list_num_frame = []

            for p in self.video_files:
                num_frames = len(os.listdir(os.path.join(self.videos_frames_root_path, p.split('.')[0])))

                for i in range(0, num_frames - self.alternate * self.len_snippet, int(self.len_snippet)):

                    self.list_num_frame.append((p, i))

    def _spatial_transform(self, clip):
        if self.spatial_transform is not None:
            clip = [self.spatial_transform(img) for img in clip]
        
        aug_info = None
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
        return clip, aug_info

    def __getitem__(self, index):

        if self.mode == 'train':
            video_name = self.video_names[index]
            video_file = self.video_files[index]

        if self.mode == 'val':
            video_file, start_frame = self.list_num_frame[index]
            video_name = video_file.split('.')[0]

        num_frames = len(os.listdir(os.path.join(self.videos_frames_root_path, video_name)))

        if(self.mode == 'train'):
            start_frame = np.random.randint(0, num_frames-self.alternate * self.len_snippet+1)

        clip = []
        for i in range(self.len_snippet):
            img = Image.open(os.path.join(os.path.join(self.videos_frames_root_path, video_name), 'img_%05d.jpg' % (start_frame+self.alternate*i+1))).convert('RGB')
            
            clip.append(img)

        clip , aug_info = self._spatial_transform(clip)

        # GETTING THE BOUNDING BOXES FOR THE CLIP
        bbox_path_video = os.path.join(self.bbox_root_path, video_name)
        key_frame_number = start_frame + self.len_snippet // 2
        bbox_path = os.path.join(bbox_path_video, '%05d.npy' % (key_frame_number+1))
        target = np.load(bbox_path).tolist()
        

        # Getting the ground truth saliency map
        clip_gt = []
        path_annt = os.path.join(self.gt_sal_maps_path, video_name, 'maps')
        
        for i in range(self.len_snippet):
            gt = np.array(Image.open(os.path.join(path_annt, 'eyeMap_%05d.jpg' % (
						start_frame+self.alternate*i+1))).convert('L'))
            gt = gt.astype('float')

            gt = cv2.resize(gt, (448, 224))

            if np.max(gt) > 1.0:
                gt = gt / 255.0

            clip_gt.append(torch.FloatTensor(gt))
            

        return {'clip':clip , 'clip_gt':clip_gt , 'target':target , 'index':index}

    def __len__(self):
        if self.mode == 'train':
            return len(self.video_names)
        else:
            return (len(self.list_num_frame))



class MVVA_AVADataLoader(data.DataLoader):
	def __init__(self,
				 dataset,
				 batch_size=1,
				 shuffle=False,
				 sampler=None,
				 batch_sampler=None,
				 num_workers=0,
				 pin_memory=False,
				 drop_last=False,
			        ):
		super(MVVA_AVADataLoader, self).__init__(
			dataset=dataset,
			batch_size=batch_size,
			shuffle=shuffle,
			sampler=sampler,
			num_workers=num_workers,
			collate_fn=self._collate_fn,
			pin_memory=pin_memory,
			drop_last=drop_last,
		        )

	def _collate_fn(self, batch):
                #clips = [_['clip'] for _ in batch]
               # clips, pad_ratios = batch_pad(clips)
                
                #print(batch)
		# For gt clips
                clips_gt = [_['clip_gt'] for _ in batch]
                
                clips = []
                for i in range(len(batch[0]['clip'])):
                    clip, pad_ratios = batch_pad([_['clip'][i] for _ in batch])
                    clips.append(clip)
                
                ids = [_['index'] for _ in batch]
                targets = [_['target'] for _ in batch]

                output = {
			'clip': clips,
			'index': ids,
			'target': targets,
			'clip_gt': clips_gt
		}
                return clips , clips_gt , targets , ids

