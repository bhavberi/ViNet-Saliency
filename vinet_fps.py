# %%
print("Started")
import onnx
import onnxruntime as ort
import torch
from torchvision import transforms, utils

import os
import cv2
from PIL import Image
from tqdm import tqdm
import time
from model import *
# %%
model_path = '/home2/bhavberi/vinet_original_optimized.onnx'
frames_path = '/home2/bhavberi/DAVIS/JPEGImages/480p/bear/'
save_path = '/ssd_scratch/cvit/bhavberi/masks'
clip_size = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
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
    scale = Scale(resize=(256))
    params = scale.randomize_parameters()
    img_transform = transforms.Compose([
        # scale,
        transforms.Resize((224, 384)),
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

# %%
def blur(img):
    k_size = 11
    bl = cv2.GaussianBlur(img, (k_size, k_size), 0)
    return torch.FloatTensor(bl)

def img_save(tensor, fp, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0, format=None):
    # print(tensor.size())
    grid = utils.make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)

    ndarr = torch.round(grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0)).to('cpu', torch.uint8).numpy()
    ndarr = ndarr[:,:,0]
    im = Image.fromarray(ndarr)
    exten = fp.split('.')[-1]
    # print(im.size)
    if exten=="png":
        im.save(fp, format=format)
    else:
        im.save(fp, format=format, quality=100) #for jpg

def bulk_save(output_tensor, paths, img_size):

    # Convert the output tensor to a numpy array on the CPU
    output_array = output_tensor.cpu().float().numpy()

    print(output_tensor.size(0))

    # Loop over the frames in the output tensor
    for i in range(output_tensor.size(0)):

        # Get the i-th frame from the output tensor
        frame = output_array[i]

        # Resize and blur the frame
        frame = cv2.resize(frame, (img_size[0], img_size[1]))
        frame = blur(frame)

        print(paths)
        # Generate a filename for the output frame
        filepath = paths[i]

        # Save the output frame as an image
        img_save(frame, filepath, normalize=True)

# %%
onnx_model = onnx.load(model_path)
onnx.checker.check_model(onnx_model, full_check=True)
print("ONNX Checked Successfully")

# %%
session_options = ort.SessionOptions()
session_options.intra_op_num_threads = 4

session = ort.InferenceSession(model_path, providers=[
    ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}),
    "CPUExecutionProvider"
  ], sess_options=session_options)
input_name = session.get_inputs()[0].name
print(input_name)

# %%

rootgrouped = False
model = VideoSaliencyModel(
        use_upsample=True,
        num_hier=3,
        num_clips=32,
        grouped_conv=rootgrouped,
        root_grouping=rootgrouped,
        depth=False,
        efficientnet=False,
        BiCubic = True,
    )

model = model.to(device)

# %%
list_frames  = [f for f in os.listdir(frames_path)]
list_frames.sort()

print("No of frames:",len(list_frames))
# list_frames = list_frames[:100]

os.makedirs(save_path, exist_ok=True)

# %%
start_time = time.time()
times = []

# %%
# for frame in tqdm(list_frames):
# process in a sliding window fashion
if len(list_frames) >= 2*clip_size-1:
    snippet = []

    output_tensor = torch.empty((0,), device='cuda')
    paths = list()
    for i in range(len(list_frames)):
        torch_img, img_size = torch_transform(
            os.path.join(frames_path, list_frames[i])
        )
        snippet.append(torch_img)

        if i >= clip_size-1:
            # if i==clip_size-1:
            #     start_time = time.time()
            start_time1 = time.time()

            clip = torch.tensor(
                torch.stack(snippet, dim=0), device=device).unsqueeze(0)
            clip = clip.permute((0, 2, 1, 3, 4))

            # with torch.no_grad():
            #     output = model(clip)
            # output = session.run(
            #     None, {input_name: clip.cpu().numpy()})[0]
            output = model(clip)
            
            output_tensor = torch.cat(
                (output_tensor, torch.tensor(output, device=device).squeeze(0)), dim=0)
            
            # print(os.path.join(save_path, list_frames[i]))
            # img = cv2.resize(output, (img_size[0], img_size[1]))
            # img = blur(output)
            # img_save(img, os.path.join(save_path, list_frames[i]), normalize=True)
            paths.append(os.path.join(save_path, list_frames[i]))
            # break
        
            # process first (len_temporal-1) frames
            if i < 2*clip_size-2:
                # output = session.run(
                #     None, {input_name: clip.cpu().numpy()})[0]
                output = model(clip)
                output_tensor = torch.cat(
                    (output_tensor, torch.tensor(output, device=device).squeeze(0)), dim=0)
                # img = cv2.resize(output, (img_size[0], img_size[1]))
                # img = blur(output)
                # img_save(img, os.path.join(save_path, list_frames[i-clip_size+1]), normalize=True)
                paths.append(os.path.join(save_path, list_frames[i-clip_size+1]))
            
            del snippet[0]
            end_time1 = time.time()

            times.append(end_time1-start_time1)

        if output_tensor.size(0) >= 300:
            # bulk_save(output_tensor, paths, img_size)

            output_tensor = torch.empty(
                (0,), dtype=torch.half, device=device)
            paths = list()
        
        
    if output_tensor.size(0) >= 0:
        # bulk_save(output_tensor, paths, img_size)

        output_tensor = torch.empty(
            (0,), dtype=torch.half, device=device)
        paths = list()
else:
    print("Not enough frames in the folder")
    # break

# %%
end_time = time.time()

print("Time taken:", end_time-start_time)
print("FPS:", len(list_frames)/(end_time-start_time))
print(1/min(times), min(times))
# %%
# output = session.run(None, {'input_name': input_data})


