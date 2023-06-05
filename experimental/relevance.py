import tqdm
from transformers import AutoTokenizer, AutoConfig
import transformers
from transformers.models.auto.modeling_auto import (
    MODEL_WITH_LM_HEAD_MAPPING_NAMES,
)
import torch
from torch.nn import CrossEntropyLoss
from torch.nn.functional import normalize
from torch import Tensor
import numpy as np
from typing import List
from torch.utils.data import DataLoader
from tqdm import tqdm

class QGen:
    
    def __init__(self, model_name:str, device: str) -> None:
        
        config = AutoConfig.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = '[PAD]'
        architecture = np.intersect1d(list(MODEL_WITH_LM_HEAD_MAPPING_NAMES.values()), config.architectures)
        self.model = getattr(transformers, architecture[0]).from_pretrained(model_name)
        self.model.eval()
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._target_device = torch.device(device)

        self.model.to(device)
        self.max_len = 512
        self.question_template = "Generate a question for the given passage\n"
        self.loss_fun = CrossEntropyLoss(reduction="none")
        self.collate_fn = self.encoder_decoder_collatefn if self.model.config.is_encoder_decoder else self.decoder_only_collatefn
    
    def decoder_only_collatefn(self, batch):
        
        query, passage = zip(*batch)
        inputs = self.tokenizer([self.question_template] * len(passage), 
                                    passage).input_ids
        labels = self.tokenizer(query).input_ids
        for i in range(len(batch)):
            if len(inputs[i] + labels[i])  > self.max_len:
                inputs[i] = inputs[i][:self.max_len]
                
        input_ids = [input+label for input,label in zip(inputs,labels)]
        batch_max_len = max(len(input) for input in input_ids)
        input_ids = self.tokenizer.pad({"input_ids":input_ids}, max_length=batch_max_len, return_tensors="pt")["input_ids"]
        mask_tensor = torch.zeros(*input_ids.shape)
        for i in range(input_ids.shape[0]):
            mask_tensor[i,:len(inputs[i])] = 1
            mask_tensor = torch.where(input_ids==self.tokenizer.pad_token_id,torch.tensor(1.0),mask_tensor)
        label_ids = torch.where(mask_tensor.bool(), 
                                torch.tensor(-100), input_ids)
        
        return {"input_ids":input_ids},label_ids
    
    def encoder_decoder_collatefn(self, batch):
        
        query, passage = zip(*batch)
        inputs = self.tokenizer([self.question_template] * len(passage), 
                                    passage, max_length=self.max_len, padding=True,
                                    truncation=True,return_tensors="pt",add_special_tokens=True)
        labels = self.tokenizer(query, max_length=self.max_len,
                                truncation=True,padding=True, return_tensors="pt").input_ids
        labels = torch.where(labels==self.tokenizer.pad_token_id,torch.tensor(-100),labels)

        return {"input_ids":inputs.input_ids,
                "attention_mask":inputs.attention_mask,
                "decoder_input_ids":labels},labels
    
    def get_loss(self,logits:Tensor, labels: Tensor):
        
        if not self.model.config.is_encoder_decoder:
            logits = logits[:,:-1,:].contiguous()
            labels = labels[:,1:].contiguous()
        losses = self.loss_fun(logits.view(-1,logits.shape[-1]),labels.view(-1)).reshape(*labels.shape)
        losses = [ 1 - normalize(loss[loss!=0],dim=-1).mean(dim=-1).item() for loss in losses]
        return np.round(losses,3)
            
    def predict(self, sentences:List[List[str]], batch_size:int = 32, show_progress:bool=True):
        
        predictions = []
        dataloader = DataLoader(sentences,batch_size=batch_size,collate_fn = self.collate_fn)
        
        if show_progress:
            dataloader = tqdm(dataloader)
        
        for i, data in enumerate(dataloader):
            inputs, labels = data
            with torch.no_grad():
                logits = self.model(**inputs,output_hidden_states=False).logits
                loss = self.get_loss(logits, labels)
                predictions.append(loss)
                
        return np.hstack(predictions)
        
        