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
from collections.abc import Iterable, Callable, Sequence
from typing import Any, Optional, Union
import yaml

from numpy import ndarray
from PIL.Image import Image
import torch
from torchvision import transforms
import pytorch_lightning as pl
from tqdm import tqdm

from parseq.tokenizer import CharsetAdapter, Tokenizer
from parseq.model import PARSeq as Model

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
    
    def forward(self, images: Union[ndarray, Image, list[ndarray|Image]], threshold: Optional[int] = None, max_length: Optional[int] = None) -> tuple[list[str], list[torch.Tensor]]:
        """Pass images through the model.

        Args:
            images (Union[ndarray, Image, list[ndarray | Image]]): Images to be passed through the model
            threshold (Optional[int], optional): Value to filter characters. If the character's probability produced by the model is lower than this value, it will be removed from the output. Defaults to None.
            max_length (Optional[int], optional): Maximum output length. If not provided, model's default max output length will be used.

        Returns:
            tuple[list[str], list[torch.Tensor]]: A list of output string and a list of confidence of each characters. Each item in the aforementioned lists corresponds to an image passed through the model.
        """
        if isinstance(images, list):
            images = [self.transforms(image) for image in images]
            images = torch.stack(images)
            single_image = False
        else:
            images = self.transforms(images).unsqueeze(0)
            single_image = True
        with torch.no_grad():
            logits = self.model.forward(self.tokenizer, images.to(self.device), max_length)
            preds = logits.softmax(-1)
            labels, confidences = self.tokenizer.decode(preds)
        if threshold:
            labels = [''.join([v[0] for v in zip(label, confidence) if v[1] > threshold]) for label, confidence in zip(labels, confidences)]
            confidences = [[v[1] for v in zip(label, confidence) if v[1] > threshold] for label, confidence in zip(labels, confidences)]
        if single_image: # Single image inference
            label = labels[0]
            confidence = confidences[0]
            return label, confidence
            
        return list(zip(labels, confidences))
    
    @classmethod
    def get_model(self, variant='parseq', state_dict_file: str = None, device = 'cpu'):
        """Return a parseq model defined by the variant.

        Args:
            variant (str, optional): Either 'parseq', 'parseq-tiny' or path to the config file. Defaults to 'parseq'.
            state_dict_file (str, optional): Path to the file containing model's state dict. Defaults to None, which means randomly initialize the model weight.
            device (str, optional): Specify the device to load the model to. Defaults to 'cpu'.

        Returns:
            PARSeq: model with the given config and state dict.
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
     
    def get_transforms(self, img_size):
        return transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(img_size, transforms.InterpolationMode.BICUBIC),
                transforms.Normalize(0.5, 0.5),
            ])
        
    def auto_batch(self, iterable: Iterable, batch_size: int = 64, preprocess_fn: Callable = None, postprocess_fn: Callable = None, threshold: Optional[int] = None, max_length: Optional[int] = None, verbose: bool = True) -> Any:
        """Automatically batch items from iterable and pass through the model.

        Args:
            iterable (Iterable): Iterable containing object to be batched
            batch_size (int, optional): Maximum batch size. Default to 64.
            preprocess_fn (Callable, optional): Function that will be applied to each item of iterable before passing the batch through the model. Defaults to None.
            postprocess_fn (Callable, optional): Function that will be applied to each item of output after passing the batch through the model. Defaults to None.
            threshold (Optional[int], optional): Value to filter characters. If the character's probability produced by the model is lower than this value, it will be removed from the output. Defaults to None.
            max_length (Optional[int], optional): Maximum output length. If not provided, model's default max output length will be used.
            verbose (bool): Whether to display process bar. Default to True
        Returns:
            Any: Items after passed through the model and post-processed.
        """
        cummu_res = []
        batch = []
        item_idx = 0
        iterable_len = len(iterable)
        for idx, item in enumerate(tqdm(iterable, disable=not verbose)):
            item_idx += 1
            if len(batch) < batch_size:
                batch.append(item)
                # After putting the item to batch, if batch size is still smaller than maximum and iterable has more item, continue batching.
                if (len(batch) < batch_size) and ((idx + 1) < iterable_len):
                    continue
            
            if preprocess_fn is not None:
                batch = [preprocess_fn(batch_item) for batch_item in batch]
            output = self.forward(batch, threshold=threshold, max_length=max_length)
            if postprocess_fn is not None:
                output = [postprocess_fn(batch_item) for batch_item in output]
            
            cummu_res.extend(output)
            batch = []
        return cummu_res
    
    @staticmethod
    def ckpt2pth(ckpt_path: str|Path, pth_path: str|Path):
        """Convert checkpoint file (`.ckpt`) produced by Pytorch Lightning to standard `.pth` file.

        Args:
            ckpt_path (str | Path): Path to checkpoint file
            pth_path (str | Path): Where to save the `.pth` file
        """
        parseq = PARSeq.load_from_checkpoint(ckpt_path).eval().cpu()
        torch.save(parseq.state_dict(), pth_path)