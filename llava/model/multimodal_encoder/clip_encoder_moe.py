import torch
import torch.nn as nn

from transformers import CLIPImageProcessor
from .configuration_clip import CLIPVisionConfig
from .modeling_clip import CLIPVisionModel

import safetensors.torch

class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.cfg = CLIPVisionConfig.from_pretrained(self.vision_tower_name)
        self.tune_mm_mlp_adapter = getattr(args, 'tune_mm_mlp_adapter')

        if not delay_load:
            self.load_model()
        
    def init_gate_weight(self):
        """Initialize the gate weights"""
        for i in range(self.cfg.num_hidden_layers):
            nn.init.normal_(self.vision_tower.vision_model.encoder.layers[i].mlp.gate.weight, std=0.01)

    def load_MoE_Blocks_weight(self):
        """
        MoE's mlp module has been modified, which will cause the weights of four vision encoder's mlp parts not be loaded
        So manually define a function to implement mlp module load
        """
        original_pretrain_dict = torch.load("data/pretrained/clip-vit-large-patch14-336/pytorch_model.bin", map_location=torch.device('cpu'))
        mlp_pretrain_dict = {k.split('mlp.')[0] + 'mlp.experts.0.' + k.split('mlp.')[1] : v for k, v in original_pretrain_dict.items() if 'mlp'in k and 'vision_model' in k }
        self.vision_tower.load_state_dict(mlp_pretrain_dict, strict=False)

        original_pretrain_dict = torch.load("data/pretrained/dinov2-large/pytorch_model.bin", map_location=torch.device('cpu'))
        mlp_pretrain_dict = {'vision_model.' + k.split('mlp.')[0][:13] + 's' + k.split('mlp.')[0][13:] + 'mlp.experts.1.' + k.split('mlp.')[1] : v for k, v in original_pretrain_dict.items() if 'mlp'in k }
        self.vision_tower.load_state_dict(mlp_pretrain_dict, strict=False)

        original_pretrain_dict = torch.load("data/pretrained/sam-vit-large/pytorch_model.bin", map_location=torch.device('cpu'))
        mlp_pretrain_dict = {'vision_model.' + (k.split('mlp.')[0] + 'mlp.experts.2.' + k.split('mlp.')[1])[7:] : v for k, v in original_pretrain_dict.items() if 'mlp'in k and 'vision_encoder' in k }
        self.vision_tower.load_state_dict(mlp_pretrain_dict, strict=False)

        original_pretrain_dict = torch.load("data/pretrained/eva02_large_patch14_clip_336.merged2b_s6b_b61k/open_clip_pytorch_model.bin", map_location=torch.device('cpu'))
        mlp_pretrain_dict = {'vision_model.encoder.layers.' + k.split('mlp.')[0].split('visual.trunk.blocks.')[1] + 'mlp.experts.3.' + k.split('mlp.')[1] : v for k, v in original_pretrain_dict.items() if 'mlp' in k and 'visual' in k }
        self.vision_tower.load_state_dict(mlp_pretrain_dict, strict=False)
        # del original_pretrain_dict
        # del mlp_pretrain_dict


    def load_pretrain_stage_MoE(self):
        # include gate and MoE parameters
        # 这个checkpoints是pretrain后的权重所在的路径
        para_dict = torch.load("checkpoints/llava-v1.5-13b-pretrain/mm_projector.bin", map_location=torch.device('cpu'))
        para_dict = {k.split('model.vision_tower.vision_tower.')[1] : v for k, v in para_dict.items() if 'model.vision_tower.vision_tower.' in k}
        self.vision_tower.load_state_dict(para_dict, strict=False)

    def load_finetune_for_eval(self, eval_load_path):
        # 权重全部存在checkpoints/llava-v1.5-13b/model-00006-of-00006.safetensors下了
        # 后续可以多写一个判断条件，来判断是finetune还是eval，这样不用手动的切换注释代码
        para_dict = safetensors.torch.load_file(eval_load_path + "/model-00006-of-00006.safetensors")
        para_dict = {k.split('model.vision_tower.vision_tower.')[1] : v for k, v in para_dict.items() if 'model.vision_tower' in k}
        self.vision_tower.load_state_dict(para_dict)
        del para_dict

    def load_model(self, eval_load_path=None):
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)        
        
        if eval_load_path is not None:
            self.load_finetune_for_eval(eval_load_path)
        else:
            if self.tune_mm_mlp_adapter is True:
                # pretrain stage
                self.init_gate_weight()
                self.load_MoE_Blocks_weight()
            else:
                # finetune stage
                self.load_pretrain_stage_MoE()

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
