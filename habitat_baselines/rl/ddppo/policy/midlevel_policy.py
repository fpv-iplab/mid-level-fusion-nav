#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import time
import torch.nn.functional as F

from habitat_baselines.common.utils import CategoricalNet, Flatten
from habitat_baselines.rl.ddppo.policy import resnet
from habitat_baselines.rl.ddppo.policy.running_mean_and_var import (
    RunningMeanAndVar,
)
from habitat_baselines.rl.models.rnn_state_encoder import RNNStateEncoder
from habitat_baselines.rl.ppo import Net, Policy
import torchvision.transforms
import torchvision.transforms.functional as TF
import visualpriors


class PointNavMidLevelPolicy(Policy):
    def __init__(
        self,
        observation_space,
        action_space,
        goal_sensor_uuid="pointgoal_with_gps_compass",
        hidden_size=512,
        num_recurrent_layers=2,
        rnn_type="LSTM",
        feature_maps=["normal"],
        encoder="default",
        device="cpu",
    ):
        super().__init__(
            PointNavMidLevelNet(
                observation_space=observation_space,
                action_space=action_space,
                goal_sensor_uuid=goal_sensor_uuid,
                hidden_size=hidden_size,
                num_recurrent_layers=num_recurrent_layers,
                rnn_type=rnn_type,
                feature_maps=feature_maps,
                encoder=encoder,
                device=device,
            ),
            action_space.n,
        )


class PointNavMidLevelNet(Net):
    """Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        goal_sensor_uuid,
        hidden_size,
        num_recurrent_layers,
        rnn_type,
        feature_maps,
        encoder,
        device,
    ):
        super().__init__()
        self.goal_sensor_uuid = goal_sensor_uuid

        self.prev_action_embedding = nn.Embedding(action_space.n + 1, 32)
        self._n_prev_action = 32

        self._n_input_goal = (
            observation_space.spaces[self.goal_sensor_uuid].shape[0] + 1
        )
        self.tgt_embeding = nn.Linear(self._n_input_goal, 32)
        self._n_input_goal = 32

        self._hidden_size = hidden_size

        rnn_input_size = self._n_input_goal + self._n_prev_action
        if encoder == "simple":
            self.visual_encoder = FeaturesEncoder(feature_maps, device)
        elif encoder == "SE_attention":
            self.visual_encoder = SE_encoder(feature_maps, device)
        elif encoder == "mid_fusion":
            self.visual_encoder = midFusion(feature_maps, device)
        self.visual_fc = nn.Sequential(
            Flatten(),
            nn.Linear(
                np.prod(self.visual_encoder.output_shape), hidden_size
            ),
            nn.ReLU(True),
        )            
        self.state_encoder = RNNStateEncoder(
            (0 if self.is_blind else self._hidden_size) + rnn_input_size,
            self._hidden_size,
            rnn_type=rnn_type,
            num_layers=num_recurrent_layers,
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def get_tgt_encoding(self, observations):
        goal_observations = observations[self.goal_sensor_uuid]
        goal_observations = torch.stack(
            [
                goal_observations[:, 0],
                torch.cos(-goal_observations[:, 1]),
                torch.sin(-goal_observations[:, 1]),
            ],
            -1,
        )

        return self.tgt_embeding(goal_observations)

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        x = []
        if not self.is_blind:
            if "visual_features" in observations:
                visual_feats = observations["visual_features"]
            else:
                visual_feats = self.visual_encoder(observations)

            visual_feats = self.visual_fc(visual_feats)
            x.append(visual_feats)

        tgt_encoding = self.get_tgt_encoding(observations)
        prev_actions = self.prev_action_embedding(
            ((prev_actions.float() + 1) * masks).long().squeeze(-1)
        )

        x += [tgt_encoding, prev_actions]

        x = torch.cat(x, dim=1)
        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)

        return x, rnn_hidden_states


#---------ENCODERS-----------------------------------------------------------------------------
class midFusion(nn.Module):
    def __init__(self, feature_maps, device):
        super().__init__()
        self.feature_maps = feature_maps
        self.device = device
        self.n_features = len(feature_maps)
        self.encoders = nn.ModuleList([])
        for feature_map in feature_maps:
            self.encoders.append(FeaturesEncoder(feature_maps=[feature_map], device=self.device).to(self.device))
        
        self.flatten = nn.Flatten()
        self.output_shape = np.prod([self.encoders[0].output_shape]) * self.n_features
    
    @property
    def is_blind(self):
        return False

    def forward(self, observations):
        encoder_embeddings = []
        for encoder in self.encoders:
            x = encoder(observations)
            encoder_embeddings.append(self.flatten(x))
        x = torch.cat(encoder_embeddings, dim=1)
        return x


class FeaturesEncoder(nn.Module):
    def __init__(
        self,
        feature_maps,
        device,
    ):
        super().__init__()
        self.feature_maps = feature_maps
        self.device = device
        # EXTRACT THE REQUIRED FEATUREMAPS. INPUT REQUIRED: LIST OF MAPS. IF > 1 MAPS, STACK THEM
        self.n_features = len(feature_maps)
        h_w_feature_map_size = 4
        out_size = 128
        channels_per_feature = 8
        input_size = channels_per_feature * self.n_features
        
        self.fusion = nn.Sequential(
            nn.Conv2d(
                input_size,
                out_size // 2,
                kernel_size = 3,
                padding = 1,
                stride = 2,
                bias = False,
            ),
            nn.GroupNorm(num_groups=4, num_channels=out_size // 2),
            nn.ReLU(True),
            nn.Conv2d(
                out_size // 2,
                out_size,
                kernel_size = 3,
                padding = 1,
                stride = 2,
                bias = False,
            ),
            nn.GroupNorm(num_groups=8, num_channels=out_size),
            nn.ReLU(True),
        )
        self.output_shape = (
            out_size,
            h_w_feature_map_size,
            h_w_feature_map_size,
        ) #(128, 4, 4)
        self.running_mean_and_var = RunningMeanAndVar(input_size)
        self.layer_init()

        self.midlevel_dicts = [dict() for _ in range(self.n_features)] # list of dicts
        self.check_dict = True

    @property
    def is_blind(self):
        return False

    def layer_init(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    layer.weight, nn.init.calculate_gain("relu")
                )
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)
    
    def get_midlevel_repr(self, obs, feature_type):
        obs = obs.unsqueeze_(0).permute(0, 3, 1, 2) # from [B x H x W x C] to [B x C x H x W]
        obs = ((obs / 255.0)*2 -1).to(self.device) # [-1,1] domain
        featuremap = visualpriors.representation_transform(obs, feature_type, device=self.device)
        return featuremap

    def forward(self, observations):
        if self.is_blind:
            return None

        rgb_observations = observations["rgb"] # [BATCH x HEIGHT X WIDTH x CHANNEL]
        global_gps = observations["global_gps"].cpu().numpy()
        global_compass = observations["global_compass"].cpu().numpy()
        representation = []
        for i, img in enumerate(rgb_observations):
            img_repr = []
            dict_key = str(round(global_gps[i][0],1))+str(round(global_gps[i][1],2))+str(round(global_compass[i][0],2))+str(round(global_compass[i][1],2))+str(round(global_compass[i][2],2))+str(round(global_compass[i][3],2))
            
            if dict_key in self.midlevel_dicts[0] and self.check_dict:
                for feat_idx in range(len(self.feature_maps)):
                    img_repr.append( self.midlevel_dicts[feat_idx][dict_key].to(self.device) )
            else:
                for feat_idx,feature in enumerate(self.feature_maps):
                    o_t = ((img / 255.0)*2 -1).to(self.device) # [-1,1] domain
                    o_t = o_t.unsqueeze_(0).permute(0, 3, 1, 2) # from [B x H x W x C] to [B x C x H x W]
                    featuremap = visualpriors.representation_transform(o_t, feature, device=self.device)
                    img_repr.append(featuremap)
                    # add extracted midlevel repr. in the midlevel repr. dictionary
                    if self.check_dict:
                        self.midlevel_dicts[feat_idx][dict_key] = featuremap.cpu()
            
            if self.check_dict and len(self.midlevel_dicts[0].keys()) > 1200000: # limit the dict elements to a given threshold
                self.check_dict = False

            representation.append( torch.cat(img_repr, dim=1) ) # from [#F, 1, C, H, W] to [1, #F*C, H, W]
        x = torch.cat(representation) # representation should a tensor of size [BATCH x CHANNEL x HEIGHT X WIDTH]
        x = self.running_mean_and_var(x)
        x = self.fusion(x)
        return x

#-------Squeeze-Excite-------------------------------------------------------------
class SE_encoder(nn.Module):
    def __init__(self, feature_maps, device, first_layer_att_only=False):
        super().__init__()
        self.feature_maps = feature_maps
        self.device = device
        self.first_layer_att_only = first_layer_att_only
        self.n_features = len(feature_maps)
        h_w_feature_map_size = 4
        out_size = 128
        channels_per_feature = 8
        input_size = channels_per_feature * self.n_features

        self.attention_1 = SELayer(input_size)
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                input_size,
                out_size // 2,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.GroupNorm(num_groups=4, num_channels=out_size // 2),
            nn.ReLU(True)
        )
        self.attention_2 = SELayer(n_input_channel=64)
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                out_size // 2,
                out_size,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.GroupNorm(num_groups=8, num_channels=out_size),
            nn.ReLU(True)
        )
        self.attention_3 = SELayer(n_input_channel=out_size)
        
        self.output_shape = (
            out_size,
            h_w_feature_map_size,
            h_w_feature_map_size,
        ) # (128, 4, 4)
        self.running_mean_and_var = RunningMeanAndVar(input_size)
        self.layer_init()

        self.midlevel_dicts = [dict() for _ in range(self.n_features)] # list of dicts
        self.check_dict = True

    @property
    def is_blind(self):
        return False

    def layer_init(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    layer.weight, nn.init.calculate_gain("relu")
                )
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    def forward(self, observations):
        if self.is_blind:
            return None

        rgb_observations = observations["rgb"] # [BATCH x HEIGHT X WIDTH x CHANNEL]
        global_gps = observations["global_gps"].cpu().numpy()
        global_compass = observations["global_compass"].cpu().numpy()
        representation = []

        for i, img in enumerate(rgb_observations):
            img_repr = []
            dict_key = str(round(global_gps[i][0],1))+str(round(global_gps[i][1],2))+str(round(global_compass[i][0],2))+str(round(global_compass[i][1],2))+str(round(global_compass[i][2],2))+str(round(global_compass[i][3],2))
            
            if dict_key in self.midlevel_dicts[0]:
                for feat_idx in range(len(self.feature_maps)):
                    img_repr.append( self.midlevel_dicts[feat_idx][dict_key].to(self.device) )
            else:
                o_t = ((img / 255.0)*2 -1).to(self.device) # [-1,1] domain
                o_t = o_t.unsqueeze_(0).permute(0, 3, 1, 2) # from [B x H x W x C] to [B x C x H x W]
                for feat_idx,feature in enumerate(self.feature_maps):
                    featuremap = visualpriors.representation_transform(o_t, feature, device=self.device)
                    img_repr.append(featuremap)
                    # add extracted midlevel repr. in the midlevel repr. dictionary
                    if self.check_dict:
                        self.midlevel_dicts[feat_idx][dict_key] = featuremap.cpu()
            
            if self.check_dict and len(self.midlevel_dicts[0].keys()) > 1200000: # limit the dict elements to a given threshold
                self.check_dict = False
            
            representation.append( torch.cat(img_repr, dim=1) ) # from [#F, 1, C, H, W] to [1, #F*C, H, W]
        x = torch.cat(representation) # representation should a tensor of size [BATCH x CHANNEL x HEIGHT X WIDTH]
        x = self.running_mean_and_var(x)
        x = self.attention_1(x)
        x = self.conv1(x)
        if not self.first_layer_att_only:
            x = self.attention_2(x)
        x = self.conv2(x)
        if not self.first_layer_att_only:
            x = self.attention_3(x)
        return x

class SELayer(nn.Module):
    def __init__(self, n_input_channel, reduction=1):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(n_input_channel, n_input_channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(n_input_channel // reduction, n_input_channel, bias=False),
            nn.Sigmoid()
        )
        self.layer_init()

    def layer_init(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    layer.weight, nn.init.calculate_gain("relu")
                )
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)
    
    def forward(self, x):
        b, c, _, _ = x.size() # e.g. batch=4, channels=16
        y = self.avg_pool(x) # [B, C, 1, 1]
        y = y.view(b, c) # [B, C]
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)