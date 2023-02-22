import pandas as pd
import numpy as np
import re
import json
import pickle

import spacy
from spacy import displacy
nlp = spacy.load("en_core_web_sm")

import networkx as nx
from networkx import NetworkXNoPath

from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def split_entity(s):
    ts=s
    sentence=""
    epos=[]
    seps=['<e1>', '</e1>', '<e2>', '</e2>']
    for i, sep in enumerate(seps):
        splitted=s.split(sep, maxsplit=1)
        if i==1 or i==3:
            start=len(sentence.split())
            epos.append([start, start+len(splitted[0].split())-1])
            #if len(splitted[0].split())>1:
            #    print(ts)
        sentence+=' '+splitted[0]
        s=splitted[1]
    sentence+=' '+splitted[1]
    token=sentence.split()
    sentence=' '.join(token)
    return {'sentence': sentence, 'epos': epos, 'token': token}


def create_relative_distance(sentence_data):
    e1_distance=[]
    e2_distance=[]
    e1_pos=sentence_data['epos'][0]
    e2_pos=sentence_data['epos'][1]
    for i in range(len(sentence_data['token'])):
        if i<e1_pos[0]:
            e1_distance.append(i-e1_pos[0])
        elif i>e1_pos[1]:
            e1_distance.append(i-e1_pos[1])
        else: 
            e1_distance.append(0)
        if i<e2_pos[0]:
            e2_distance.append(i-e2_pos[0])
        elif i>e2_pos[1]:
            e2_distance.append(i-e2_pos[1])
        else: 
            e2_distance.append(0)
    return e1_distance, e2_distance


def dependency_parsing_edge(sentence):
    # nlp function returns an object with individual token information, linguistic features and relationships
    doc = nlp(sentence)
    edges=[]
    splitted=[]
    for token in doc:
        #splitted.append(str(token.text))
        #x=[[str(token.text), str(token.head.text), str(token.dep_)]]
        #x.append([token.i, token.head.i, str(token.dep_)])
        #edges.append(x)
        edges.append([token.i, token.head.i, str(token.dep_), token.text])
    return edges


def matrix_from_edges(edges, n, initial_val):
    matrix=[[initial_val]*n for i in range(n)]
    for edge in edges:
        matrix[edge[0]][edge[1]]=edge[2]
    return matrix


def get_shortest_path(edges, a, b):
    edges=[[x[0], x[1]] for x in edges]
    und_graph = nx.Graph(edges)
    try:
        und_path=nx.shortest_path(und_graph, source=a, target=b)
    except NetworkXNoPath:
        return [], None
    else:
        edges=[edges[i] for i in und_path]
        di_graph=nx.DiGraph(edges)
        for node in und_path:
            try:
                di_path1=nx.shortest_path(di_graph, source=a, target=node)
                di_path2=nx.shortest_path(di_graph, source=b, target=node)
            except NetworkXNoPath:
                pass
            else:
                return und_path, node
        return [], None


def path_between_2entity(sentence_data, edges):
    first_e1_pos=sentence_data['epos'][0][0]
    first_e2_pos=sentence_data['epos'][1][0]
    path, grand_parent=get_shortest_path(edges, first_e1_pos, first_e2_pos)
    path_array=['N']*len(edges)
    try:
        grand_parent_pos=path.index(grand_parent)
    except ValueError:
        pass
    else:
        for i, node in enumerate(path):
            path_array[node]=i-grand_parent_pos
    finally:
        return path_array

def sentence_bump(data):
    regex=''
    try: 
        data['sentence']=data['sentence'].apply(lambda x: ' '.join(re.findall('<e1>*|<\/e1>*|<e2>*|<\/e2>*|\w*', x)))
        data['sentence']=data['sentence'].apply(lambda x: re.sub('cannot', 'can not', x))
        data['sentence']=data['sentence'].apply(lambda x: re.sub('gonna', 'gon na', x))
        data['sentence']=data['sentence'].apply(lambda x: re.sub('theres', 'there is', x))
        data2=list(data.apply(lambda row: split_entity(row['sentence']), axis=1))
    except:
        data=[' '.join(re.findall('<e1>*|<\/e1>*|<e2>*|<\/e2>*|[a-zA-Z]*', x)) for x in data]
        data2=[split_entity(x) for x in data]
    return data2 #list of dict


def get_feature(data):
    data2=sentence_bump(data)

    e1_distance, e2_distance, grammar, shortest_path=[], [], [], []
    for sentence_data in data2:
        #distant
        distance=create_relative_distance(sentence_data)
        e1_distance.append(distance[0])
        e2_distance.append(distance[1])

        edges=dependency_parsing_edge(sentence_data['sentence'])
        grammar.append(edges)

        shortest_path.append(path_between_2entity(sentence_data, edges))
        
    sentences=[x['sentence'] for x in data2]
    return sentences, e1_distance, e2_distance, grammar, shortest_path


class RE_DataEncoder():
    def __init__(self, vocab_size, max_len, sentences_train, grammar_train, label_train):
        self.max_len=max_len
        self.vocab_size=vocab_size
    
        #for the sentences
        self.tokenizer = Tokenizer(num_words=vocab_size, filters='!"#$%&()*+,-./:;=?@[]^_`{|}~', lower=True, oov_token=1)
        self.tokenizer.fit_on_texts(sentences_train)
        self.word_index = self.tokenizer.word_index
        self.word_size = len(self.word_index) #different from vocab size

        #for the grammar
        grammar_type=[]
        for edges in grammar_train:
            grammar_type.extend([x[2] for x in edges])
        grammar_type=list(set(grammar_type))
        self.grammar2idx={v:i+1 for i, v in enumerate(grammar_type)}

        #for the label
        self.lbencoder = LabelEncoder().fit(label_train)
        self.dict_labels={i: w for i, w in enumerate(list(self.lbencoder.classes_))}


    def encode(self, sentences, e1_distance, e2_distance, grammar, shortest_path):
        #sentence
        sequences = self.tokenizer.texts_to_sequences(sentences)
        sentences_encode = pad_sequences(sequences, maxlen=self.max_len, value=0, padding='post')

        #relative distance
        e1_distance=pad_sequences(e1_distance, maxlen=self.max_len, value=999, padding='post')
        e2_distance=pad_sequences(e2_distance, maxlen=self.max_len, value=999, padding='post')
        e1_distance+=self.max_len
        e2_distance+=self.max_len
        e1_distance[e1_distance==999+self.max_len]=0
        e2_distance[e2_distance==999+self.max_len]=0

        #grammar relantion
        grammar_matrix=[]
        for edge_list in grammar:
            n=len(edge_list)
            for i in range(len(edge_list)):
                try:
                    edge_list[i][2]=self.grammar2idx[edge_list[i][2]]
                except:
                    #edge_list[i][2]=self.grammar2idx['ROOT']
                    edge_list[i][2]=0
            matrix=[x[2] for x in edge_list]
            grammar_matrix.append(matrix)

        grammar_matrix=pad_sequences(grammar_matrix, maxlen=self.max_len, value=0, padding='post')

        #shortest path
        for i in range(len(shortest_path)):
            for j in range(len(shortest_path[i])):
                if shortest_path[i][j]=='N':
                    shortest_path[i][j]=1
                else:
                    shortest_path[i][j]+=self.max_len+1
        
        shortest_path=pad_sequences(shortest_path, maxlen=self.max_len, value=0, padding='post')
    
        return sentences_encode, e1_distance, e2_distance, grammar_matrix, shortest_path

    def encode_label(self, label_name):
        label = self.lbencoder.transform(label_name)
        label = to_categorical(label, num_classes=len(set(label_name))) # from keras.utils.np_utils
        label = np.array(label)
        return label

# data=pd.read_csv('/content/drive/MyDrive/nlp/relation_extraction/data/kbp37-master.csv')
# data=data.dropna()
# data=data.rename(columns={'sentences': 'sentence'})
# data['relationship']=data['relationship'].str.strip()

# def drop_sentence_no_ent(row):
#     sentence_data=split_entity(row['sentence'])
#     if sentence_data['epos'][0][1]<sentence_data['epos'][0][0] or sentence_data['epos'][1][1]<sentence_data['epos'][1][0]:
#         row['sentence']=None
#     return row

# data=data.apply(drop_sentence_no_ent, axis=1).dropna()

# train, test = train_test_split(data, test_size=0.2)

# tokenizer = Tokenizer(num_words=1, filters='!"#$%&()*+,-./:;=?@[]^_`{|}~', lower=True, oov_token=1)
# tokenizer.fit_on_texts(kk[0])
# word_index = tokenizer.word_index
# word_size = len(word_index)
# word_size

# train.to_csv('/content/drive/MyDrive/nlp/relation_extraction/data/train.csv')
# test.to_csv('/content/drive/MyDrive/nlp/relation_extraction/data/test.csv')

# nlp = spacy.load("en_core_web_sm")
# data = pd.read_csv('/content/drive/MyDrive/nlp/relation_extraction/data/'+'train'+'.csv')
# data=data[:10]
# sentences, e1_distance, e2_distance, grammar, shortest_path, label=get_feature(data)

# with open('/content/drive/MyDrive/nlp/relation_extraction/data/'+'train'+'_features.json', 'r') as r:
#                 data=json.load(r)
# sentences=data['sentences']
# e1_distance=data['e1_distance']
# e2_distance=data['e2_distance']
# dependency_direct=data['dependency_direct']
# dependency_type=data['dependency_type']
# label=data['label']

# import seaborn as sns
# print(max([len(x) for x in dependency_type]))
# sns.histplot([len(x) for x in dependency_type]);

if __name__ == "__main__":

    vocab_size=20000
    max_len = 50
    read_new_data=False


    for type in ['train', 'test']:
        if read_new_data==True:
            data = pd.read_csv('data/'+type+'.csv')
            sentences, e1_distance, e2_distance, grammar, shortest_path=get_feature(data)
            label=list(data['relationship'])
            with open('data/'+type+'_features.json', 'w') as w:
                json.dump({'sentences': sentences, 
                        'e1_distance': e1_distance, 
                        'e2_distance': e2_distance, 
                        'grammar': grammar,
                        'shortest_path': shortest_path,
                        'label': label
                        }, w)     
        else:
            with open('data/'+type+'_features.json', 'r') as r:
                data=json.load(r)
            sentences=data['sentences']
            e1_distance=data['e1_distance']
            e2_distance=data['e2_distance']
            grammar=data['grammar']
            shortest_path=data['shortest_path']
            label=data['label']

        if type=='train':
            Encoder = RE_DataEncoder(vocab_size, max_len, sentences, grammar, label)

        sentences_np, e1_distance_np, e2_distance_np, grammar_np, shortest_path_np = Encoder.encode(sentences, 
                                                                                                    e1_distance, 
                                                                                                    e2_distance, 
                                                                                                    grammar, 
                                                                                                    shortest_path,
                                                                                                    )
        label_np=Encoder.encode_label(label)
                                                                                          
        all_features=np.array([sentences_np, e1_distance_np, e2_distance_np, grammar_np, shortest_path_np])
        np.save('data/X_'+type+'.npy', all_features)
        np.save('data/y_'+type+'.npy', label_np)

        #save encoder class
        with open('data/data_encoder.obj', 'wb') as f:
            pickle.dump(Encoder, f)