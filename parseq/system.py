# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from typing import Any, Optional, Sequence, Union
import yaml

from numpy import ndarray
from PIL.Image import Image
import torch
from torchvision import transforms
import pytorch_lightning as pl

from parseq.tokenizer import CharsetAdapter, Tokenizer

from .model import PARSeq as Model

CONFIG_DIR = Path('./parseq/configs')

class PARSeq(pl.LightningModule): # Maintain the use of pytorch lightning for future conversion from ckpt file after training to pth file for inference

    def __init__(
        self,
        max_label_length: int,
        img_size: Sequence[int],
        patch_size: Sequence[int],
        embed_dim: int,
        enc_num_heads: int,
        enc_mlp_ratio: int,
        enc_depth: int,
        dec_num_heads: int,
        dec_mlp_ratio: int,
        dec_depth: int,
        decode_ar: bool,
        refine_iters: int,
        dropout: float,
        charset: str=None,
        charset_train: str = None,
        charset_test: str = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if charset:
            self.charset=charset
        else:
            assert charset_train == charset_test, "Only support the same train and test charset."
            self.charset = charset_train
        self.tokenizer = Tokenizer(self.charset) # TODO: just 1 charset -> when convert from ckpt, need to convert (`charset_train`, `charset_test`) to `charset`
        self.charset_adapter = CharsetAdapter(self.charset)
        self.transforms = self.get_transforms(img_size)
        self.save_hyperparameters()

        self.model = Model(
            len(self.tokenizer),
            max_label_length,
            img_size,
            patch_size,
            embed_dim,
            enc_num_heads,
            enc_mlp_ratio,
            enc_depth,
            dec_num_heads,
            dec_mlp_ratio,
            dec_depth,
            decode_ar,
            refine_iters,
            dropout,
        )
    
    def forward(self, images: Union[ndarray, Image, list[ndarray|Image]], threshold: Optional[int] = None, max_length: Optional[int] = None) -> torch.Tensor:
        if isinstance(images, list):
            images = [self.transforms(image) for image in images]
            images = torch.stack(images)
        else:
            images = self.transforms(images).unsqueeze(0)
        with torch.no_grad():
            logits = self.model.forward(self.tokenizer, images.to(self.device), max_length)
            preds = logits.softmax(-1)
            labels, confidences = self.tokenizer.decode(preds)
        if threshold:
            labels = [''.join([v[0] for v in zip(label, confidence) if v[1] > threshold]) for label, confidence in zip(labels, confidences)]
            confidences = [[v[1] for v in zip(label, confidence) if v[1] > threshold] for label, confidence in zip(labels, confidences)]
        return labels, confidences
    
    @classmethod
    def get_model(self, variant='parseq', state_dict_file: str = None, device = 'cpu'):
        """Return a parseq model defined by the variant.

        Args:
            variant (str, optional): Either 'parseq', 'parseq-tiny' or path to the config file. Defaults to 'parseq'.
        """
        if variant in ['parseq', 'parseq-tiny']:
            config_path = CONFIG_DIR/f"{variant}.yml"
        else:
            config_path = variant
        with open(config_path, mode='r', encoding='utf-8') as f:
            init_config = yaml.safe_load(f)
        model = PARSeq(**init_config)
        if state_dict_file:
            state_dict = torch.load(state_dict_file, weights_only=True)
            model.load_state_dict(state_dict)
        return model.eval().to(device)
    
    @classmethod
    def ckpt2pth(self, ckpt_path, pth_path):
        parseq = PARSeq.load_from_checkpoint(ckpt_path).eval().cpu()
        torch.save(parseq.state_dict(), pth_path)
        
    def get_transforms(self, img_size):
        return transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(img_size, transforms.InterpolationMode.BICUBIC),
                transforms.Normalize(0.5, 0.5),
            ])