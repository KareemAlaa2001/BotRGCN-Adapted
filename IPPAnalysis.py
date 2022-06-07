from Dataset import Twibot20
# import torch
import pandas as pd

import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector

def get_lang_detector(nlp, name):
    return LanguageDetector()

print('Loading dev.json')
df_dev=pd.read_json('./Twibot-20/dev.json')
print('Loading train.json')
df_train=pd.read_json('./Twibot-20/train.json')
print('Loading test.json')
df_test=pd.read_json('./Twibot-20/test.json')

# print('Loading support.json')
# df_support=pd.read_json('./Twibot-20/support.json')   

print('Finished Loading Dataframes')


def examineDataframes():
    print('Examining Dataframes')
    print('Columns:')
    print(df_dev.columns)

    print('\n\n')
    print('Contents of tweet for index 1')
    # print(df_dev.iloc[1]['tweet'])
    # print(type(df_dev.iloc[1]['tweet']))
    
    print('Number of nodes in train.json:')
    print(len(df_train))
   
    print('Number of nodes in test.json:')
    print(len(df_test))

    print('Number of nodes in dev.json:')
    print(len(df_dev))
    
    
    pass

def langDetectDataset():
    # print('Loading train.json')
    # df_train=pd.read_json('./Twibot-20/train.json')
    # print('Loading test.json')
    # df_test=pd.read_json('./Twibot-20/test.json')
    # print('Loading support.json')
    # df_support=pd.read_json('./Twibot-20/support.json')
    nlp = spacy.load('en_core_web_sm')
    Language.factory("language_detector", func=get_lang_detector)
    nlp.add_pipe('language_detector', last=True)
    
    # df_combined = pd.concat([df_train, df_test, df_dev, df_support])
    print('testing lang detection on test.json')
    df_test['lang'] = df_test['tweet'].apply(lambda x: pd.Series(map(lambda y: nlp(y)._.language, x)) if x else None)
    print("Language value count result for df_test:")
    print(df_test['lang'].value_counts())

    # print('Language value count result for combined')
    # print(df_combined['tweet'].apply(lambda x: pd.Series(x)).apply(lambda x: nlp(x)._.language).stack().value_counts())
    # print(df_combined['lang'].value_counts())


   
    


def basicLangDetectExample():
    nlp = spacy.load('en_core_web_sm')
    Language.factory("language_detector", func=get_lang_detector)
    nlp.add_pipe('language_detector', last=True)
    text = 'This is an english text.'
    text2 = 'Ich bin ein deutscher Text.'
    text3 = "Je suis un texte en français."
    doc = nlp(text)
    doc2 = nlp(text2)
    doc3 = nlp(text3)
    # document level language detection. Think of it like average language of the document!
    print(doc._.language)
    # sentence level language detection
    for sent in doc.sents:
        print(sent, sent._.language)

     # document level language detection. Think of it like average language of the document!
    print(doc2._.language)
    # sentence level language detection
    for sent in doc2.sents:
        print(sent, sent._.language)
    
     # document level language detection. Think of it like average language of the document!
    print(doc3._.language)
    # sentence level language detection
    for sent in doc3.sents:
        print(sent, sent._.language)

# examineDataframes()
langDetectDataset()
    

