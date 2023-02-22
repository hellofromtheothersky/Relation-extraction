import re
import pickle
from relation_extraction.process_data import get_feature, RE_DataEncoder
from keras.models import model_from_json

MODEL_NAME='cnn'
BASE_DIR='relation_extraction/'

def load_model(model_name):
    path = BASE_DIR+"saved_model/"
    with open(path + model_name + ".json", "r") as rf:
        jstr = rf.read()
    model = model_from_json(jstr)
    model.load_weights(path + model_name + ".h5")
    return model


def go_predict(model, input):
    with open(BASE_DIR+"data/data_encoder.obj", "rb") as rf:
        Encoder = pickle.load(rf)

    sentences, e1_distance, e2_distance, grammar, sp = get_feature([input])
    sentences_np, e1_dist_np, e2_dist_np, grammar_np, sp_np = Encoder.encode(
        sentences, e1_distance, e2_distance, grammar, sp
    )
    print(sentences, sp_np)

    pred = model.predict([sentences_np, e1_dist_np, e2_dist_np, grammar_np, sp_np])
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

    label, acc = go_predict(load_model(MODEL_NAME), text)
    re_type=label.split('(')[0]
    if 'e1,e2' in label:
        return e1, re_type, e2, round(acc*100, 1)
    else:
        return e2, re_type, e1, round(acc*100, 1)


if __name__ == "__main__":
    print(
        go_predict(
            load_model(MODEL_NAME),
            'the <e1>Titanic</e1> set sail on its maiden voyage, traveling from Southampton, England, to <e2>New York City</e2>',
        )
    )
