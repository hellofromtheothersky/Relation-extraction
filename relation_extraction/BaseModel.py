from abc import ABC, abstractmethod
from sklearn.metrics import classification_report
BASE_DIR='saved_model/'
class BaseModel(ABC):
    @abstractmethod
    def build_model(self):
        pass
        

    def predict(self, X):
        return self.model.predict([X_i for X_i in X])


    def save_model(self, model_name):
        model_structure=self.model.to_json()
        with open(BASE_DIR+model_name+'.json', 'w') as json_file:
            json_file.write(model_structure)
        self.model.save_weights(BASE_DIR+model_name+'.h5')

    def load_model(self, model_name):
        from keras.models import model_from_json
        with open(BASE_DIR+model_name+'.json', "r") as rf:
            jstr=rf.read()
        model=model_from_json(jstr)
        model.load_weights(BASE_DIR+model_name+'.h5')
        self.model=model

    def train_model(self, X, y, epochs=1):
        self.model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
        history = self.model.fit([X_i for X_i in X], 
                    y, 
                    epochs=epochs,
                    batch_size=32,
                    validation_split=0.1)
        

    def evaluate(self, X_test, y_test, dict_labels):
        preds = self.predict(X_test)
        print('Full classes:')
        print(classification_report(y_test.argmax(axis=1), preds.argmax(axis=1), zero_division=0, target_names=list(dict_labels.values())))
        print('Excluding Other and Entity-Destination(e2,e1):')
        print(classification_report(y_test.argmax(axis=1), preds.argmax(axis=1), 
                            target_names=[x[1] for x in dict_labels.items() if x[1]!='Other' and x[1]!='Entity-Destination(e2,e1)'], 
                            labels=[x[0] for x in dict_labels.items() if x[1]!='Other' and x[1]!='Entity-Destination(e2,e1)']))
        #import matplotlib.pyplot as plt
        #plt.figure(figsize=(20, 7))
        #sns.heatmap(confusion_matrix(y_test.argmax(axis=1), preds.argmax(axis=1)), annot=True, fmt=".1f")