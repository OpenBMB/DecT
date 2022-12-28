import os, shutil
import sys
sys.path.append(".")

import torch
from torch import nn
from torch.nn.parallel.data_parallel import DataParallel
from openprompt.utils.cuda import model_to_device
from tensorboardX import SummaryWriter

from tqdm import tqdm
import dill
import warnings

from typing import Callable, Union, Dict
try:
    from typing import OrderedDict
except ImportError:
    from collections import OrderedDict
from sklearn.metrics import accuracy_score
from openprompt.pipeline_base import PromptForClassification
from openprompt import PromptDataLoader
from openprompt.prompts import *
from openprompt.utils.logging import logger

class DecTRunner(object):
    r"""A runner for DecT
    This class is specially implemented for classification.

    Args:
        model (:obj:`PromptForClassification`): One ``PromptForClassification`` object.
        train_dataloader (:obj:`PromptDataloader`, optional): The dataloader to bachify and process the training data.
        valid_dataloader (:obj:`PromptDataloader`, optionla): The dataloader to bachify and process the val data.
        test_dataloader (:obj:`PromptDataloader`, optional): The dataloader to bachify and process the test data.
        config (:obj:`CfgNode`): A configuration object.
        loss_function (:obj:`Callable`, optional): The loss function in the training process.
    """
    def __init__(self, 
                 model: PromptForClassification,
                 train_dataloader: Optional[PromptDataLoader] = None,
                 valid_dataloader: Optional[PromptDataLoader] = None,
                 test_dataloader: Optional[PromptDataLoader] = None,
                 calibrate_dataloader: Optional[PromptDataLoader] = None,
                 loss_function: Optional[Callable] = None,
                 id2label: Optional[Dict] = None,
                 ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.calibrate_dataloader = calibrate_dataloader
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.id2label = id2label
        self.label_path_sep = config.dataset.label_path_sep   
        self.clean = True
    
    def inference_step(self, batch, batch_idx):
        label = batch.pop('label')
        logits = self.model(batch)
        pred = torch.argmax(logits, dim=-1)
        return pred.cpu().tolist(), label.cpu().tolist()
    
    def inference_epoch(self, split: str): 
        outputs = []
        self.model.eval()
        with torch.no_grad():
            data_loader = self.valid_dataloader if split=='validation' else self.test_dataloader
            for batch_idx, batch in enumerate(data_loader):
                batch = batch.cuda().to_dict()
                outputs.append( self.inference_step(batch, batch_idx) )

        score = self.inference_epoch_end(outputs)
        logger.info(f"{split} Performance: {score}")
        return score 

    def inference_epoch_end(self, outputs):
        preds = []
        labels = []
        for pred, label in outputs:
            preds.extend(pred)
            labels.extend(label)

        score = accuracy_score(preds, labels)
        return score

    def training_step(self, batch, batch_idx):
        logits = self.model(batch)
        loss = self.loss_function(logits, batch['label'])
        return loss

    def on_fit_start(self):
        """Some initialization works"""
        self.inner_model.verbalizer.train_proto(self.model, self.train_dataloader, self.calibrate_dataloader, self.config.environment.local_rank)

    def fit(self, ckpt: Optional[str] = None):
        self.set_stop_criterion()
        self.configure_optimizers()

        if ckpt:
            if not self.load_checkpoint(ckpt):
                logger.warning("Train from scratch instead ...")

        self.inner_model.verbalizer.train_proto(self.model, self.train_dataloader, self.calibrate_dataloader, self.config.environment.local_rank)
        
        score = self.inference_epoch("validation")

        return score
    
    def test(self, ckpt: Optional[str] = None) -> dict:
        if ckpt:
            if not self.load_checkpoint(ckpt, load_state = False):
                logger.error("Test cannot be performed")
                exit()
        return self.inference_epoch("test")

    def run(self, ckpt: Optional[str] = None) -> dict:
        self.fit(ckpt)
        return self.test(ckpt = None if self.clean else 'best')