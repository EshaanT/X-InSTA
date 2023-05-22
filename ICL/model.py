
import numpy as np
import os
import torch
import torch.nn.functional as F

from tqdm import tqdm
from transformers import Adafactor, AdamW, get_linear_schedule_with_warmup
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


class ICLModel(object):

    def __init__(self, logger=None,mode='eval'):

        self.logger = logger

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()

        self.n_gpu = n_gpu
        self.device = device
        self.model = None
        self.tokenizer=None
        self.mode = mode

    def __str__(self):
        text = "[ICL Model]: "
        if self.model_name is None:
            text += "No model loaded yet"
        else:
            text += self.model_name
            if self.mode is None:
                text += " (no mode setted - try .train() or .eval()"
            else:
                text += " (%s mode)" % self.mode
        text += "\nusing device %s, %d gpus, local_rank=%d" % (self.device, self.n_gpu, self.local_rank)
        return ("="*50) + "\n" + text + "\n" + ("="*50)

    def is_none(self):
        return self.model is None

    def train(self):
        self.model.train()
        self.mode = "train"

    def eval(self):
        self.model.eval()
        self.mode = "eval"

    def cuda(self):
        self.model.cuda()

    def to_device(self):
        self.model.to(self.device)

    def load(self, model_name="facebook/xglm-564M"):
        '''
        checkpoint can be either keyword of the model or path to the checkpoint file
        '''

        
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        self.model = model
        self.tokenizer=AutoTokenizer.from_pretrained(model_name)




    def do_inference(self, data, batch_size=1, verbose=True):
        dataloader = data.get_dataloader(batch_size, is_training=False)
        if verbose:
            dataloader = tqdm(dataloader, desc='Infering the test set')
        losses = []
        for batch in dataloader:
            input_ids=batch[0].cuda()
            attention_mask=batch[1].cuda()
            token_type_ids=batch[2].cuda()
            if len(batch)==3:
                labels=None
            else:
                labels=batch[3].cuda()
            with torch.no_grad():
                loss = self.run_model(input_ids, attention_mask, token_type_ids, labels=labels)
            losses += loss.cpu().detach().numpy().tolist()
        return losses

    def do_predict(self, data, batch_size=1, losses=None, verbose=True):
        if losses is None:
            losses = self.do_inference(data, batch_size, verbose=verbose)
        losses = np.array(losses)
        assert len(losses)==len(data)
        predictions = []
        for idx, dp in enumerate(data.metadata):
            curr_label_losses = [np.sum(losses[indices]) for indices in dp["indices"]]
            #print(curr_label_losses)
            prediction_idx = sorted(enumerate(curr_label_losses), key=lambda x: x[1])[0][0]
            prediction = dp["options"][prediction_idx]
            predictions.append(prediction.strip())
        return predictions


    def run_model(self, input_ids, attention_mask, token_type_ids, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        logits = outputs.logits[..., :-1, :].contiguous()
        if labels is None:
            labels = input_ids
        labels = labels[..., 1:].contiguous()
        label_mask = token_type_ids[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        losses = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1)) # [batch_size, length]

        losses = losses.view(logits.size(0), logits.size(1)) * label_mask
        return torch.sum(losses, axis=1) / torch.sum(label_mask, axis=1)


