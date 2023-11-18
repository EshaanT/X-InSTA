import os

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, util


def get_logic(tar,src,dataset,flip,simple=False):

    lang_code={
        'en':{
        'en':'english',
        'de':'german',
        'fr':'french',
        'es':'spanish',
        'zh':'mandarin',
        'ja':'japaneese',
        'jp':'japaneese',
        'pt':'portugese',
        'it':'italian',
        'hi':'hindi'},

        'pt':{
        'pt':'português',
        'it':'italiana',
        'hi':'hindi',
        'en':'inglês',
        'de':'alemã',
        'fr':'francesa',
        'es':'espanhola'
        },

        'it':{
        'pt':'portoghese',
        'it':'italiana',
        'hi':'hindi',
        'en':'inglese',
        'de':'tedesca',
        'fr':'francese',
        'es':'spagnola'
        },
        
        'hi':{
        'pt':'पुर्तगाली',
        'it':'इतालवी',
        'hi':'हिंदी',
        'en':'अंग्रेज़ी',
        'de':'जर्मन',
        'fr':'फ्रेंच',
        'es':'स्पैनिश'
        },

        'de':{
        'en':'englisch',
        'de':'deutsche',
        'fr':'französisch',
        'es':'spanisch',
        'zh':'Mandarin',
        'ja':'japanisch',
        'jp':'japanisch',
        'pt':'Portugiesisch',
        'it':'italian',
        'hi':'hindi'},

        'fr':{
        'en':'Anglaise',
        'de':'Allemande',
        'fr':'Français',
        'es':'Espagnole',
        'zh':'mandarine',
        'ja':'japonaise',
        'jp':'japonaise',
        'pt':'Portugais',
        'it':'italienne',
        'hi':'hindi'},

        'es':{
        'en':'Inglesa',
        'de':'Alemana',
        'fr':'francesa',
        'es':'Española',
        'zh':'mandarín',
        'ja':'japonesa',
        'jp':'japonesa',
        'pt':'Portuguesa',
        'it':'italiana',
        'hi':'hindi'},

        'zh':{
        'en':'英语',
        'de':'德语',
        'fr':'法语',
        'es':'西班牙语',
        'zh':'普通话',
        'ja':'日本人',
        'jp':'日本人'},

        'ja':{
        'en':'英語',
        'de':'ドイツ人',
        'fr':'フランス語',
        'es':'スペイン語',
        'zh':'マンダリン',
        'ja':'日本',
        'jp':'日本'},
        
        'jp':{
        'en':'英語',
        'de':'ドイツ人',
        'fr':'フランス語',
        'es':'スペイン語',
        'zh':'マンダリン',
        'ja':'日本',
        'jp':'日本'},


    }
    
    amaz_bi_logic={
        'en':'In {lo} bad means {bo} and good means {go}.',
        'de':'In {lo} bedeutet schlecht {bo} und gut bedeutet {go}.',
        'fr':'Dans {lo} bien signifie {bo} et mal signifie {go}.',
        'es':'En {lo} malo significa {bo} y bueno significa {go}.',
        'zh':'在 {lo} 坏手段 {bo} 和好的手段 {go}.',
        'ja':'の {lo} 悪い 手段 {bo} と 良い 手段 {go}.'
    }
    
    hateval_logic={
        'en':'In {lo} no hate means {bo} odio and yes hate means {go} odio.',
        'es':'En {lo} no odio significa {bo} hate y sí odio significa {go} hate.'
    }
    
    cls_logic={
        'en':'In {lo} bad means {bo} and good means {go}.',
        'de':'In {lo} bedeutet schlecht {bo} und gut bedeutet {go}.',
        'fr':'Dans {lo} bien signifie {bo} et mal signifie {go}.',
        'es':'En {lo} malo significa {bo} y bueno significa {go}.',
        'jp':'の {lo} 悪い 手段 {bo} と 良い 手段 {go}.'
    }

    amaz_bi_logic_s={
        'en':'The following post is in {lo}.',
        'de':'der folgende Beitrag ist in {lo}',
        'fr':"le post suivant c'est dans {lo}",
        'es':'la siguiente publicación está en {lo}.',
        'zh':'以下帖子发布在 {lo}',
        'ja':'次の投稿は {lo} に投稿されました'
    }

    cls_logic_s={
        'en':'The following post is in {lo}.',
        'de':'der folgende Beitrag ist in {lo}',
        'fr':"le post suivant c'est dans {lo}",
        'es':'la siguiente publicación está en {lo}.',
        'jp':'次の投稿は {lo} に投稿されました'
    }

    logic=''

    if dataset=='amaz_bi':

        logic=amaz_bi_logic[src]

        if simple:
            logic=amaz_bi_logic_s[src]
        

        if flip==True:
            return logic.format(lo=lang_code[src][tar],go=amaz_bi_v[tar]['negative'],bo=amaz_bi_v[tar]['positive'])

        
        return logic.format(lo=lang_code[tar][tar],bo=amaz_bi_v[tar]['negative'],go=amaz_bi_v[tar]['positive'])

    elif dataset=='cls':
        return cls_logic[src].format(lo=lang_code[src][tar],bo=cls_v[tar]['negative'],go=cls_v[tar]['positive'])
    elif dataset=='hateval':
        return hateval_logic[src].format(lo=lang_code[src][tar],bo=hateval_v[tar][0],go=hateval_v[tar][1])
        
    else:
        raise ValueError('Logic not coded')


def uniform_label(tar,src,test):

    tar_d=amaz_bi_v[tar]
    src_d=amaz_bi_v[src]

    options=list(src_d.values())

    print('Old options were ',test[0]['options'])
    print('New options are ',options)
    tar_d={v: k for k, v in tar_d.items()}
    
    for t in test:
        t['options']=options
        t['output']=src_d[tar_d[t['output']]]
    
    return test


task2lang={
    'amaz_bi':['de', 'en', 'es', 'fr', 'ja', 'zh'],
    'hateval':['en','es'],
    'cls':['en','de','fr','jp']
}
hateval_v={
    'en':{
        1:'yes',
        0:'no'
    },
    'es':{
        1:'sí',
        0:'no'
    }
}

amaz_bi_v={
    'en':{
        'negative':'bad',
        'positive':'good'
    },
    'fr':{
        'negative':'mal',
        'positive':'bien'
    },
    'es':{
        'negative':'malo',
        'positive':'bueno'
    },
    'ja':{
        'negative':'悪い',
        'positive':'良い'
    },
    'zh':{
        'negative':'坏的',
        'positive':'好的'
    },
    'de':{
        'negative':'Schlecht',
        'positive':'gut'
    }
}
cls_v={
    'en':{
        'negative':'bad',
        'positive':'good'
    },
    'fr':{
        'negative':'mal',
        'positive':'bien'
    },
    'jp':{
        'negative':'悪い',
        'positive':'良い'
    },
    'de':{
        'negative':'Schlecht',
        'positive':'gut'
    }
}

"""def sample_from_dataframe(df,k,seed):
    p_k = k // len(df['output'].unique())

    df_train_k = df.groupby('output', group_keys=False).apply(lambda df: df.sample(p_k, random_state=seed))
    _df_train = df[~df['id'].isin(df_train_k['id'])].reset_index(drop=True)

    need = k - len(df_train_k)
    need_df = _df_train.sample(need)

    return pd.concat([df_train_k, need_df]).reset_index(drop=True)"""

def sample_from_dataframe(df,k,seed):

    df_train_k=df.sample(k, random_state=seed)

    return df_train_k.reset_index(drop=True)
 
def input_form_converter(dataset_name,test_df,demo_df=[]):
    
    if type(demo_df)!=list:
        demos = demo_df.to_dict(orient='records')
    else:
        demos=demo_df
    test = test_df.to_dict(orient='records')

    """add options and demos in test set"""
    for d in test:
        d['options']=get_options(d,dataset_name)  
        d['demos']=demos
    return test

def get_options(df,dataset_name):

    op=[]
    if dataset_name=='amaz_bi':
        op=list(amaz_bi_v[df['language']].values())
    elif dataset_name=='hateval':
        op=list(hateval_v[df['language']].values())
    elif dataset_name=='cls':
        op=list(cls_v[df['language']].values())
    else:
        raise ValueError('No options')  
    
    return op

def save_file(test_final,dataset_name,l,set_up,src,s,k):
    "SAVING FILE"
    file_path='data/processed/{}/{}'.format(dataset_name,l)
    os.makedirs(file_path, exist_ok=True)

    print('Saving Files')

    file_name=os.path.join(file_path,'{}_src={}_tar={}_s={}_k={}.pkl'.format(set_up,src,l,s,k))

    print(f'Saving s {s} tar {l} src {src}')

    with open(file_name, 'wb') as handle:
        pickle.dump(test_final, handle, protocol=pickle.HIGHEST_PROTOCOL)

def create_few_shots(dataset_name='amaz_bi',src_l=None,k=16,seeds=[32,5,232,100,42],set_up='src_is_cross'):


    """

    :param dataset_name: name of the csv file
    :param src_l: a list of languages to sample from, if None passed we sample from all languages other than target
    :param k: number of samples to take default 16
    :param seeds: 5 seeds default to [32,5,232,100,42]
    :param set_up: name of the demonstration sampling technique
    :return: Saves a json list of dictionary of the form [ {'input':text,'demonstrations': {dict of k-shots of text-demonstration pairs}, 'output': label}]
    """

    train_set=f'data/processed/{dataset_name}_train.csv'
    test_set = f'data/processed/{dataset_name}_test.csv'

    if set_up in ['sim_in_cross']:
        from sentence_transformers import SentenceTransformer, util
        import torch
        embedder = SentenceTransformer('distiluse-base-multilingual-cased-v1')


    for s in seeds:
        df_train = pd.read_csv(train_set)
        df_test = pd.read_csv(test_set)

        print(f'For seed {s} and dataset {dataset_name} crating {k} few shot of {set_up} set_up')

        languages=task2lang[dataset_name]

        for l in languages:

            if set_up=='random':

                src='all'

                """Sample demonstration from all language other than l"""
                df_train_l=df_train[~df_train['language'].isin([l])].reset_index(drop=True)
                df_test_l=df_test[df_test['language'].isin([l])].reset_index(drop=True)

                """Making sure we get same number of label of each kind"""

                demo_df=sample_from_dataframe(df_train_l,k,seed=s)
                assert len(demo_df) == k

                """Converting into our standard form"""
                test_final=input_form_converter(dataset_name,df_test_l,demo_df)
                save_file(test_final,dataset_name,l,set_up,src,s,k)           

            elif set_up=='src_is_cross':
                src_l=task2lang[dataset_name].copy()
                src_l.remove(l)

                for src in src_l: 
                    df_train_l=df_train[df_train['language'].isin([src])].reset_index(drop=True)
                    df_test_l=df_test[df_test['language'].isin([l])].reset_index(drop=True)

                    """Making sure we get same number of label of each kind"""

                    demo_df=sample_from_dataframe(df_train_l,k,seed=s)
                    assert len(demo_df) == k

                    """Converting into our standard form"""
                    test_final=input_form_converter(dataset_name,df_test_l,demo_df)
                    save_file(test_final,dataset_name,l,set_up,src,s,k)     


            elif set_up=='sim_in_cross':

                src_l=task2lang[dataset_name].copy()
                src_l.remove(l)

                for src in src_l:                
                    df_train_l=df_train[df_train['language'].isin([src])].reset_index(drop=True)
                    df_test_l=df_test[df_test['language'].isin([l])].reset_index(drop=True)

                    test_final = df_test_l.to_dict(orient='records')
                    corpus_input=df_train_l['input'].to_list()
                    corpus_out=df_train_l['output'].to_list()
                    corpus=df_train_l['input'].to_list()
                    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

                    top_k=k

                    for df in test_final:
                        query=df['input']
                        query_embedding = embedder.encode(query, convert_to_tensor=True)

                        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
                        top_results = torch.topk(cos_scores, k=top_k)
                        demos=[]

                        print("\n\n======================\n\n")
                        print("Query:", query)
                        print(f"\nTop {k} most similar sentences in corpus:")
                        for score, idx in zip(top_results[0], top_results[1]):
                            print(corpus[idx], "(Score: {:.4f})".format(score))

                            demos.append(
                                {
                                    'input':corpus_input[idx],
                                    'output':corpus_out[idx],
                                    'score':score.cpu().detach().numpy().tolist(),
                                    'language':'en'
                                }
                            )
                        df['demos']=demos
                            
                        df['options']=get_options(df,dataset_name)
                    save_file(test_final,dataset_name,l,set_up,src,s,k)   

            elif set_up=='no_demo':
                src='no'

                df_test_l=df_test[df_test['language'].isin([l])].reset_index(drop=True)

                """Converting into our standard form"""
                test_final=input_form_converter(dataset_name,df_test_l,demo_df=[])
                save_file(test_final,dataset_name,l,set_up,src,s,k)

            else:
                raise ValueError('Incorect Set-up')

def load_data(dataset_name,tar,src,set_up,seed,k=16):

    if src==None:
        src='all'

    file_path='data/processed/{}/{}'.format(dataset_name,tar)
    if not os.path.exists(file_path):
        raise ValueError('We dont have data for specified target language')

    file_name = os.path.join(file_path, '{}_src={}_tar={}_s={}_k={}.pkl'.format(set_up, src, tar, seed, k))
    with open(file_name, 'rb') as handle:
        data=pickle.load(handle)

    return data

    
    
