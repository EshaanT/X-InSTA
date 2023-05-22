import os
import pandas as pd
import pickle
import argparse
import torch
import math
import string
import logging
import numpy as np
import json
from tqdm import tqdm

from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import AutoTokenizer,XGLMTokenizer
from transformers import AutoModelForCausalLM,XGLMForCausalLM
from sklearn.metrics import f1_score
import random

from ICL.data import ICLData
from ICL.model import ICLModel
from utils.data import load_data,task2lang,get_logic,uniform_label

def main(args):
    
    print("Loding Model")
    model=ICLModel()
    model.load(model_name=args.model_name)
    model.cuda()
    model.eval()

    if args.use_demonstrations:
        args.max_length= min(args.max_length_per_example * args.k, args.max_length)
        if args.set_up=='no_demo':
            raise ValueError('Use_demonstrations in no_demo setup ??')
    else:
        args.max_length=args.max_length_per_example # i.e 512
        args.set_up='no_demo'
    seeds=args.seeds.split(',')

    

    if args.tar_l==None:
        tar_l=task2lang[args.dataset_name]
    else:
        tar_l=args.tar_l.split(',')

    for s in seeds:
        for tar in tar_l:

            if args.dataset_name in ['amaz_bi','hateval','cls']:
                args.options=None
            else:
                raise ValueError('Expected options')
                
            src_l=task2lang[args.dataset_name].copy()
            print(tar)
            src_l.remove(tar) 
                
            for src in src_l:

                if args.use_logic==True:
                    logic=get_logic(tar,src,args.dataset_name,flip=args.flip,simple=args.simple_logic)
                elif args.cor_logic==True:
                     
                    cor_lang=task2lang[args.dataset_name].copy()
                    cor_lang.remove(src)
                    cor_lang.remove(tar)
                    tar_cor=random.sample(cor_lang,k=1)[0]
                    
                    print(src,tar,tar_cor)
                    
                    logic=get_logic(tar_cor,src,args.dataset_name,flip=args.flip,simple=args.simple_logic)
                else:
                    logic=None
                
                print("Evaluating tar {} src {} seed {}".format(tar,src,s))

                test_data=load_data(dataset_name=args.dataset_name,tar=tar,src=src,set_up=args.set_up,seed=s,k=args.k)

                if args.label_uniform==True:

                    test_data=uniform_label(tar,src,test_data)

                #test_data=test_data[:100]
                
                data=ICLData(model_name=args.model_name,use_demonstrations=args.use_demonstrations,use_logic=(args.use_logic or args.cor_logic) ,k=args.k,
                            max_length=args.max_length,max_length_per_example=args.max_length_per_example,method=args.method)
                data.tensorize(data=test_data,options=args.options,logic=logic,add_newlines=args.add_newlines)
                
                print(data.print_tensorized_example())
                #continue
                predictions=model.do_predict(data=data,batch_size=args.batch_size)
                groundtruths=[dp['output'] for dp in test_data]
                eval_dic=data.evaluate(predictions,groundtruths)

                if args.dataset_name in ['amaz_bi','hateval','hasoc','cls']:
                    eval_dic['f1']=f1_score(groundtruths,predictions,average='macro')
                eval_dic['seed']=s
                eval_dic['dataset_name']=args.dataset_name
                eval_dic['target_language']=tar
                eval_dic['source_language']=src
                eval_dic['set_up']=args.set_up
                eval_dic['k']=args.k
                eval_dic['logic']=(args.use_logic or args.cor_logic)
                eval_dic['model_name']=args.model_name

                set_up_name=args.set_up + ('_logic' if args.use_logic else '')

                result_path='result_{}_{}_k={}.csv'.format(args.dataset_name,set_up_name,args.k)

                if os.path.exists(result_path):
                    result_df=pd.read_csv(result_path)
                    result_df=result_df.append(eval_dic,ignore_index=True)
                else:
                    result_df=pd.DataFrame([eval_dic])

                result_df.to_csv(result_path,index=False)

                if args.model_name=='facebook/xglm-7.5B':
                    out_path=f'output/{args.dataset_name}/{args.set_up}/{tar}'
                    out_file=os.path.join(out_path,f'model=7.5Bk={args.k}s={s}.json')
                    os.makedirs(out_path,exist_ok=True)

                    
                    with open(out_file,'w') as fp:
                        json.dump(predictions,fp)
                
                print("Infrence Done for seed{} tar {} src{}".format(s,tar,src))





if __name__=='__main__':

    parser=argparse.ArgumentParser()

    parser.add_argument('--use_demonstrations',default=False,action='store_true')
    parser.add_argument('--use_logic',default=False,action='store_true')
    parser.add_argument('--label_uniform',default=False,action='store_true')

    parser.add_argument('--simple_logic',default=False,action='store_true')
    parser.add_argument('--flip',default=False,action='store_true')
    parser.add_argument('--cor_logic',default=False,action='store_true')
    parser.add_argument('--add_newlines',default=False,action='store_true')
    parser.add_argument('--dataset_name',type=str,required=True,help='The dataset_name')
    parser.add_argument('--k',type=int,default=16)
    parser.add_argument('--seeds',type=str,default='32,5,232,100,42')
    parser.add_argument('--set_up', type=str, required=True,choices=['src_is_cross','sim_in_cross'])

    parser.add_argument('--batch_size',type=int,default=8)
    parser.add_argument('--model_name',type=str,default="facebook/xglm-564M")
    parser.add_argument('--options',type=str,default=None)
    parser.add_argument('--method',type=str,default='direct')

    parser.add_argument('--max_length_per_example',type=int,default=512)
    parser.add_argument('--max_length',type=int,default=512)
    parser.add_argument('--tar_l',type=str,default=None,help='pass string of family you want to sample')
    parser.add_argument('--src_l',type=str,default=None,help='pass string of family you want to sample')
    args=parser.parse_args()
    main(args)
