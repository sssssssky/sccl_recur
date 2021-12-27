import os
from hashlib import md5
import pandas as pd
from tqdm.auto import tqdm
import random
import requests

import nlpaug.augmenter.word as naw
from nlpaug.util import Action
import time


def wordNet(train_text, label, dataset_name = ''):
    print(f"\n-----WordNet Augmenter-----\n")
    wordnet_aug = naw.SynonymAug(
        aug_src='wordnet', model_path=None, name='Synonym_Aug', aug_min=2, aug_max=10, aug_p=0.3, lang='eng')

    print("train_text:", len(train_text), type(train_text[0]))

    auglist1, auglist2 = [], []
    for txt in tqdm(train_text):
        atxt = wordnet_aug.augment(txt, n=2)
        auglist1.append(str(atxt[0]))
        auglist2.append(str(atxt[1]))
        
    train_data = pd.DataFrame()
    train_data['text'] = train_text
    train_data['label'] = label
    train_data["text1"] = pd.Series(auglist1)
    train_data["text2"] = pd.Series(auglist2)
    train_data.to_csv('{}_wordnet_augment.csv'.format(dataset_name), index=False)

    for o, a1, a2 in zip(train_text[:5], auglist1[:5], auglist2[:5]):
        print("-----Original Text: \n", o)
        print("-----Augmented Text1: \n", a1)
        print("-----Augmented Text2: \n", a2)

def Contextual(train_text, label, dataset_name = ''):
    print(f"\n-----Contextual_augment-----\n")
    augmenter1 = naw.ContextualWordEmbsAug(
        model_path='roberta-base', action="substitute", aug_min=1, aug_p=0.3, device = 'cuda')

    augmenter2 = naw.ContextualWordEmbsAug(
        model_path='bert-base-uncased', action="substitute", aug_min=1, aug_p=0.3, device = 'cuda')

    print("train_text:", len(train_text), type(train_text[0]))

    auglist1, auglist2 = [], []
    for txt in tqdm(train_text):
        atxt1 = augmenter1.augment(txt)
        atxt2 = augmenter2.augment(txt)
        auglist1.append(str(atxt1))
        auglist2.append(str(atxt2))

    train_data = pd.DataFrame()
    train_data['text'] = train_text
    train_data['label'] = label
    train_data["text1"] = pd.Series(auglist1)
    train_data["text2"] = pd.Series(auglist2)
    train_data.to_csv('{}_contextual_augment.csv'.format(dataset_name), index=False)

    for o, a1, a2 in zip(train_text[:3], auglist1[:3], auglist2[:3]):
        print("-----Original Text: \n", o)
        print("-----Augmented Text1: \n", a1)
        print("-----Augmented Text2: \n", a2)

        
def word_deletion(train_text, label, dataset_name = ''):
    ### wordnet based data augmentation
    print(f"\n-----word_deletion-----\n")
    aug = naw.RandomWordAug(aug_min=1, aug_p = 0.2)
    
    print("train_text:", len(train_text), type(train_text[0]))

    augtxts1, augtxts2 = [], []
    for txt in tqdm(train_text):
        atxt = aug.augment(txt, n=2, num_thread=1)
        augtxts1.append(str(atxt[0]))
        augtxts2.append(str(atxt[1]))
    
    train_data = pd.DataFrame()
    train_data['text'] = train_text
    train_data['label'] = label
    train_data["text1"] = pd.Series(augtxts1)
    train_data["text2"] = pd.Series(augtxts2)
    train_data.to_csv('{}_wordDelete_augment.csv'.format(dataset_name), index=False)
    
    for o, a1, a2 in zip(train_text[:3], augtxts1[:3], augtxts2[:3]):
        print("-----Original Text: \n", o)
        print("-----Augmentation1: \n", a1)
        print("-----Augmentation2: \n", a2)
        
        
def back_translation(train_text, label, dataset_name = ''):
    batch_size = 80
    batch_text = ''
    auglist1 = []
    auglist2 = []
    for i, text in tqdm(enumerate(train_text)):
        batch_text = batch_text + text +'\n'

        if( ((i+1) % batch_size == 0) & (i != 0) ):
            target_result1 = get_trans(batch_text, 'en', 'fra')
            time.sleep(2)
            target_result2 = get_trans(batch_text, 'en', 'zh')
            time.sleep(2)
            final_result1 = get_trans(get_translation_text(target_result1), 'fra', 'en')
            time.sleep(2)
            final_result2 = get_trans(get_translation_text(target_result2), 'zh', 'en')
            time.sleep(2)
            for j in range(batch_size):
                auglist1.append(final_result1[j]['dst'])
                auglist2.append(final_result2[j]['dst'])
            batch_text = ''
        

    train_data = pd.DataFrame()
    train_data['text'] = train_text
    train_data['label'] = label
    train_data["text1"] = pd.Series(auglist1)
    train_data["text2"] = pd.Series(auglist2)
    train_data.to_csv('{}_translation_augment.csv'.format(dataset_name), index=False)

    for o, a1, a2 in zip(train_text[:5], auglist1[:5], auglist2[:5]):
        print("-----Original Text: \n", o)
        print("-----Augmented Text1: \n", a1)
        print("-----Augmented Text2: \n", a2)
    

def get_translation_text(target_result):
    final_query=''
    for s in target_result:
        final_query+=s['dst']
        final_query+='\n'
        #print(final_query)
    return final_query


def get_trans(query,from_lang,to_lang):
    appid = '20211028000985394'
    appkey = 'YXIqHnk4aBOkxkLpY32c'
    endpoint = 'http://api.fanyi.baidu.com'
    path = '/api/trans/vip/translate'
    url = endpoint + path

    def make_md5(s, encoding='utf-8'):
        return md5(s.encode(encoding)).hexdigest()

    salt = random.randint(32768, 65536)
    sign = make_md5(appid + query + str(salt) + appkey)

    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    payload = {'appid': appid, 'q': query, 'from': from_lang, 'to': to_lang, 'salt': salt, 'sign': sign}
    try:
        r = requests.post(url, params=payload, headers=headers)
        result = r.json()
        return result['trans_result']
        
    except:
        time.sleep(2)
        r = requests.post(url, params=payload, headers=headers)
        result = r.json()
        return result['trans_result']


if __name__ == '__main__':
    
    files = os.listdir('../train_data/all_origin_data')
    dataset_name = [f.split('.')[0] for f in files]
    for i,f in enumerate(files):
        print("train_text:",f)
        data = pd.read_csv('../input/all-origin-data/'+f)
        train_text = data['text'].values.tolist()
        label = data['label'].values.tolist()
        back_translation(train_text, label, dataset_name = dataset_name[i])
