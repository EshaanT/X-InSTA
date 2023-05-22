
import os
import csv
import json
import string
import numpy as np
import pickle as pkl
import math
import torch

from collections import defaultdict
from functools import partial
from multiprocessing import Pool

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score

class ICLData(object):

    def __init__(self, logger=None, tokenizer=None, use_demonstrations=True,use_logic=False, k=16,
                 max_length=1024, max_length_per_example=256,model_name="facebook/xglm-564M",method='direct'
                 ):

        self.method=method
        self.logger = logger
        self.tokenizer = tokenizer
        self.use_demonstrations = use_demonstrations
        self.use_logic=use_logic
        self.k = k
        self.max_length = max_length
        self.max_length_per_example = max_length_per_example


        self.tensorized_inputs = None
        self.metadata = None

        if self.tokenizer is None:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def __len__(self):
        if self.tensorized_inputs is None:
            return 0
        return len(self.tensorized_inputs["input_ids"])

    def __str__(self):
        text = "[ICL Data]: method=%d, "
        if self.use_demonstrations:
            text += "%d demonstrations\n" % self.k
        else:
            text += "no demonstrations\n"
        if self.metadata is None:
            text += "Currently not containing any examples"
        else:
            text += "Currently containing %d examples with %d tensors to be fed in\n" % (len(self.metadata), len(self))
            text += "\n"
            text += self.print_tensorized_example(return_string=True)
        return ("="*50) + "\n" + text + "\n" + ("="*50)

    def get_dataloader(self, batch_size, is_training):
        inputs = self.tensorized_inputs
        for k, v in inputs.items():
            if type(v)==list:
                inputs[k] = torch.LongTensor(v)
        shape = inputs["input_ids"].shape
        
        for v in inputs.values():
            assert v.shape==shape
        if "labels" in inputs:
            dataset = TensorDataset(inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"], inputs["labels"])
        else:
            dataset = TensorDataset(inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"])
        if is_training:
            sampler=RandomSampler(dataset)
        else:
            sampler=SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
        return dataloader

    def evaluate(self, predictions, groundtruths, is_classification=False):
        assert len(predictions)==len(self.metadata)
        accs = []
        precisions = defaultdict(list)
        recalls = defaultdict(list)
        for prediction, groundtruth in zip(predictions, groundtruths):
            prediction = prediction.strip()
            groundtruth = [str(gt).strip() for gt in groundtruth] if type(groundtruth)==list else str(groundtruth).strip()
            is_correct = prediction in groundtruth if type(groundtruth)==list else prediction==groundtruth
            accs.append(is_correct)
            if is_classification:
                recalls[groundtruth].append(is_correct)
                precisions[prediction].append(is_correct)

        if not is_classification:
            return {'acc':np.mean(accs)}

        f1s = []
        for key in recalls:
            precision = np.mean(precisions[key]) if key in precisions else 1.0
            recall = np.mean(recalls[key])
            if precision+recall==0:
                f1s.append(0)
            else:
                f1s.append(2*precision*recall / (precision+recall))

        return {'f1':np.mean(f1s)}

    def _prepro_each_datapoint(self, dp, is_first=True, is_training=False,
                               for_demonstrations=False,add_newlines=False):
        dp = dp.copy()
        if add_newlines:
            no_label = False
            no_input = dp["input"]==""
            if self.method=="direct":
                if not is_first:
                    if no_input:
                        dp["input"] = "\n\n" + str(dp["input"])
                    else:
                        dp["input"] = "\n\n\n" + str(dp["input"])
                if not no_label:
                    dp["output"] = "\n" + str(dp["output"])
                    if "options" in dp:
                        dp["options"] = ["\n" + str(opt) for opt in dp["options"]]
            elif self.method=="channel":
                if not is_first:
                    dp["output"] = "\n\n\n" + str(dp["output"])
                    if "options" in dp:
                        dp["options"] = ["\n\n\n" + str(opt) for opt in dp["options"]]
                if not no_input:
                    if not no_label:
                        dp["input"] = "\n" + str(dp["input"])
            else:
                raise NotImplementedError()
        else:
            if not is_first:
                if self.method=="direct":
                    dp["input"] = " " + str(dp["input"])
                elif self.method=="channel":
                    dp["output"] = " " + str(dp["output"])
                    if "options" in dp:
                        dp["options"] = [" "+str(opt) for opt in dp["options"]]
                else:
                    raise NotImplementedError()
            if self.method=="direct":
                dp["output"] = " " + str(dp["output"])
                if "options" in dp:
                    dp["options"] = [" " + str(opt) for opt in dp["options"]]
            elif self.method=="channel":
                dp["input"] = " " + str(dp["input"])
            else:
                raise NotImplementedError()
        input_tokens = self.tokenizer(dp["input"])["input_ids"]

        if is_training or for_demonstrations:
            output_tokens = self.tokenizer(dp["output"],add_special_tokens=False)["input_ids"]

            if len(input_tokens)>=self.max_length_per_example - 2 - len(output_tokens):
                input_tokens = input_tokens[:self.max_length_per_example - 2 - len(output_tokens)]

            if self.method=="direct":
                return input_tokens, output_tokens
            elif self.method=="channel":
                return output_tokens, input_tokens
            else:
                raise NotImplementedError()
        else:
            assert len(dp["options"])>=2, dp

            assert dp["output"] in dp["options"]
            option_tokens = [self.tokenizer(option,add_special_tokens=False)["input_ids"] for option in dp["options"]]
            option_length = np.max([len(option) for option in option_tokens])

            if len(input_tokens)>=self.max_length_per_example - 2 - option_length:
                input_tokens = input_tokens[:self.max_length_per_example - 2 - option_length]

            input_tokens = [input_tokens for _ in option_tokens]
            output_tokens = option_tokens
            option_tokens = [dp["options"].index(dp["output"])]

            if self.method=="direct":
                return input_tokens, output_tokens, option_tokens
            elif self.method=="channel":
                return output_tokens, input_tokens, option_tokens
            else:
                raise NotImplementedError()

    def tensorize(self, data, options=None,logic=None,
                  add_newlines=False):

        """
        :param data: a list of dictionary
        :param options: list of tokens to map the output into
        :param add_newlines: add new line in demonstration output pairs
        """

        #assert type(options)==list
        assert type(data)==list

        for dp in data:
            assert type(dp)==dict
            assert 'input' in dp and 'options' in dp and 'output' in dp and 'demos' in dp

        bos_token_id=self.tokenizer.bos_token_id
        eos_token_id = self.tokenizer.eos_token_id

        input_ids,attention_mask,token_type_ids=[],[],[]
        metadata=[]

        for dp_index,dp in enumerate(tqdm(data, desc='Tokenising Data')):


            if self.use_demonstrations:

                assert len(dp['demos']) == self.k
                demonstrations=[]
                for demo_i,demo in enumerate(dp['demos']):
                    input_,output_ =self._prepro_each_datapoint(
                        demo,is_first=demo_i==0,for_demonstrations=True,
                        add_newlines=add_newlines
                    )
                    demonstrations+=input_+output_

            if self.use_logic:
                logic_ids = self.tokenizer(logic)["input_ids"]
                demonstrations+=logic_ids

            inputs,outputs,answer=self._prepro_each_datapoint(
                dp,is_first=not self.use_demonstrations
                ,add_newlines=add_newlines
            )

            indices = [[i] for i in range(len(input_ids), len(input_ids) + len(inputs))]

            metadata.append({"indices": indices, "answer": answer, "options": dp["options"]})

            for input_,output_ in zip(inputs,outputs):
                if self.use_demonstrations:
                    input_=demonstrations+input_

                encoded = prepro_sentence_pair_single(
                        input_,output_
                        ,self.max_length,
                        allow_truncation=self.use_demonstrations)

                input_ids.append(encoded[0])
                attention_mask.append(encoded[1])
                token_type_ids.append(encoded[2])

        print('Maximum sentence length: ',max(torch.LongTensor(attention_mask).sum(axis=1)))

        self.tensorized_inputs=dict(
            input_ids=torch.LongTensor(input_ids),
            attention_mask=torch.LongTensor(attention_mask),
            token_type_ids=torch.LongTensor(token_type_ids)
        )
        self.metadata=metadata



    def print_tensorized_example(self, return_string=True):
        assert self.tensorized_inputs is not None

        idx = 0
        text = "Checking the first example..."
        input_ids = self.tensorized_inputs["input_ids"][idx]
        token_type_ids = self.tensorized_inputs["token_type_ids"][idx]
        if type(input_ids)!=list:
            input_ids = input_ids.numpy().tolist()
        if type(token_type_ids)!=list:
            token_type_ids = token_type_ids.numpy().tolist()

        #print(input_ids.numpy().tolist()[0])
        text += "\nInput:\n"
        text += self.tokenizer.decode(input_ids[:token_type_ids.index(1)])
        text += "\nOutput:\n"
        text += self.tokenizer.decode([_id for _id, _type_id in zip(input_ids, token_type_ids) if _type_id==1])
        if return_string:
            return text

def prepro_sentence_pair_single(ids1, ids2, max_length,
                                allow_truncation=False):

    if allow_truncation and len(ids1)+len(ids2) > max_length:
        ids1 = ids1[len(ids1)+len(ids2)-max_length:] # len = max_length-len(ids2)
        assert len(ids1)+len(ids2)==max_length

    n_mask = max_length-len(ids1)-len(ids2)
    assert n_mask>=0, (max_length, len(ids1), len(ids2))
    input_ids = ids1+ids2+[0 for _ in range(n_mask)]
    attention_mask = [1 for _ in ids1+ids2] + [0 for _ in range(n_mask)]
    token_type_ids = [0 for _ in ids1] + [1 for _ in ids2] + [0 for _ in range(n_mask)]
    return input_ids, attention_mask, token_type_ids

def prepro_sentence_pair(train_inputs, test_inputs, max_length,
                         bos_token_id, eos_token_id,
                         allow_truncation=False):
    input_ids, attention_mask, token_type_ids = [], [], []
    for test_input in test_inputs:
        for train_input in train_inputs:
            _input_ids, _attention_mask, _token_type_ids = \
                prepro_sentence_pair_single(train_input, test_input, max_length,
                                            bos_token_id, eos_token_id,
                                            allow_truncation=allow_truncation)
            
            input_ids.append(_input_ids)
            attention_mask.append(_attention_mask)
            token_type_ids.append(_token_type_ids)
    

    return {"input_ids": torch.LongTensor(input_ids),
            "attention_mask": torch.LongTensor(attention_mask),
            "token_type_ids": torch.LongTensor(token_type_ids)}
