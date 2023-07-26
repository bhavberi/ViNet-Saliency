from modeling_finetune import VisionTransformer
import torch
from torch import nn

class VideoSaliencyModel(nn.Module):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=80,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 init_scale=0.,
                 all_frames=16,
                 tubelet_size=2,
                 use_checkpoint=False,
                 use_mean_pooling=True):
        super(VideoSaliencyModel, self).__init__()

        self.backbone = VisionTransformer(img_size = img_size , patch_size=patch_size,in_chans=in_chans,num_classes=num_classes,embed_dim=embed_dim,depth=depth,num_heads=num_heads,mlp_ratio=mlp_ratio,qkv_bias=qkv_bias,qk_scale=qk_scale,drop_rate=drop_rate,attn_drop_rate=attn_drop_rate,drop_path_rate=drop_path_rate,norm_layer=norm_layer,init_values=init_values,use_learnable_pos_emb=use_learnable_pos_emb,init_scale=init_scale,use_checkpoint=use_checkpoint,use_mean_pooling=use_mean_pooling)

        self.decoder = DecoderConvUp()
    
    def forward(self, x,boxes):
        x = self.backbone(x,boxes)
        print("After Backbone" , x.shape)
        x = self.decoder(x)
        print("After Decoder" , x.shape)
        x = x.view(x.size(0) , x.size(2) , x.size(3)) # b , 224 , 448
        return x
    

class DecoderConvUp(nn.Module):
    def __init__(self):
        super(DecoderConvUp, self).__init__()
        self.upsampling = nn.Upsample(scale_factor=(2,2), mode='bilinear')	
        self.convtsp1 = nn.Sequential(
			nn.Conv2d(768, 480, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False), # 480 , 7 , 7
			nn.ReLU(),
			self.upsampling # 480 , 14 , 14
		)
        self.convtsp2 = nn.Sequential(
			nn.Conv2d(480, 192, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False),
			nn.ReLU(), # 192 , 14 , 14
			self.upsampling # 192 , 28 , 28
		)
        self.convtsp3 = nn.Sequential(
			nn.Conv2d(192, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False),
			nn.ReLU(), # 64 , 28 , 28
			self.upsampling, # 64 , 56 , 56

			nn.Conv2d(64, 32, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False),
			nn.ReLU(), # 32 , 56 , 56
			self.upsampling, # 32 , 112 , 112

			# 4 time dimension
			nn.Conv2d(32, 32, kernel_size=(3,3), stride=(2,1),padding=(1,1), bias=False),
			nn.ReLU(), # 32 , 56 , 112
            self.upsampling, # 32 , 112 , 224  
                        
                        nn.Conv2d(32,32,kernel_size=(1,1) , stride=(1,1) , bias = False),
                        nn.ReLU(),# 32,112,224
                        self.upsampling,# 32,224,448

			nn.Conv2d(32, 1, kernel_size=1, stride=1, bias=True), # 1 , 224 , 448
                        
			nn.Sigmoid() # 1 , 224 , 448
		)
                
    def forward(self, x):
        x = self.convtsp1(x)
        print("After ConvTSP1" , x.shape)
        x = self.convtsp2(x)
        print("After ConvTSP2" , x.shape)
        x = self.convtsp3(x)
        print("After ConvTSP3" , x.shape)
        return x
