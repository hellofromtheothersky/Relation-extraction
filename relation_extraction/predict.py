import re
import pickle
from relation_extraction.process_data import get_feature_sp_based, RE_DataEncoder_sp_based
from keras.models import model_from_json
import numpy as np


def load_model(model_name):
    # path='/content/drive/MyDrive/nlp/relation_extraction/saved_model/'
    path = "relation_extraction/saved_model/"
    with open(path + model_name + ".json", "r") as rf:
        jstr = rf.read()
    model = model_from_json(jstr)
    model.load_weights(path + model_name + ".h5")
    return model


def go_predict(model, input):
    with open("relation_extraction/data_encoder.obj", "rb") as rf:
        Encoder = pickle.load(rf)

    sentences, e1_distance, e2_distance, dependency_direct, dependency_type = get_feature_sp_based([input])
    sentences, e1_distance, e2_distance, dependency_direct, dependency_type = Encoder.encode(
        sentences, e1_distance, e2_distance, dependency_direct, dependency_type
    )

    sent1=[]
    sent2=[]
    e1_1, e2_1, e1_2, e2_2 = [], [], [], []
    sp_dir, sp_type = [], []
    for i in range(1):
        sent1.append(sentences[i][:-1])
        sent2.append(sentences[i][1:])
        e1_1.append(e1_distance[i][:-1])
        e1_2.append(e1_distance[i][1:])
        e2_1.append(e2_distance[i][:-1])
        e2_2.append(e2_distance[i][1:])
        sp_dir.append(dependency_direct[i][:-1])
        sp_type.append(dependency_direct[i][:-1])

    sent1=np.array(sent1)
    sent2=np.array(sent2)
    e1_1=np.array(e1_1)
    e1_2=np.array(e1_2)
    e2_1=np.array(e2_1)
    e2_2=np.array(e2_2)
    sp_dir=np.array(sp_dir)
    sp_type=np.array(sp_type)

    pred = model.predict([sent1, sent2, e1_1, e1_2, e2_1, e2_2, sp_dir, sp_type])
    acc=pred[0].max()
    pred = pred.argmax(axis=1)
    return Encoder.dict_labels[int(pred)], acc


def sentence_token(text):
    text = re.sub(r"[^\w ]", "", text)
    text = re.sub(r"(\s+\s\s*|\s*\s\s+)", " ", text)
    text = text.strip()
    return text.split()


def predict(text, e1pos, e2pos):
    text = sentence_token(text)

    e1pos = sorted(list(set([int(x) for x in e1pos])))
    e2pos = sorted(list(set([int(x) for x in e2pos])))
    print(e1pos, e2pos)
    p1 = e1pos[0]
    p2 = e1pos[-1] + 1
    p3 = e2pos[0]
    p4 = e2pos[-1] + 1
    e1=" ".join(text[p1:p2])
    e2=" ".join(text[p3:p4])
    text = (
        " ".join(text[:p1])
        + "<e1>"
        + " ".join(text[p1:p2])
        + "</e1>"
        + " ".join(text[p2:p3])
        + "<e2>"
        + " ".join(text[p3:p4])
        + "</e2>"
        + " ".join(text[p4:])
    )
    print(text)

    label, acc = go_predict(load_model("cnn_nonbert"), text)
    re_type=label.split('(')[0]
    if 'e1,e2' in label:
        return e1, re_type, e2, round(acc*100, 1)
    else:
        return e2, re_type, e1, round(acc*100, 1)


if __name__ == "__main__":
    print(
        go_predict(
            load_model("cnn_nonbert"),
            #'<e1>he</e1> go to <e2>bed</e2>'
            #'<e1>iphone</e1> is made by <e2>apple</e2>'
            '<e1>apple</e1> make <e2>iphone</e2>'
        )
    )
