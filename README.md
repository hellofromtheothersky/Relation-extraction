# Relation-extraction

First NLP project for DS201

Presentation: https://drive.google.com/file/d/15wI1wEjx5oPcLP9qO_LwUoaUwolhMJg6/view?usp=share_link

Report: https://drive.google.com/file/d/1ev6AhwPH10waH9YuJCfRrS3pFJviFDXx/view?usp=sharing

Relation Extraction (RE) is the task of finding the relation which exists between two words (or groups of words) in a sentence. However, there are different ways of expressing the same kind of relationship. An effective solution needs to be able to account for useful semantic and syntactic features not only for the meanings of the target entities at the lexical level but also for their immediate context and for the overall sentence structure.  Shortest path dependency is used for this task to extract the most informative words that make the most influence to the type of the relationship. We also consider a feature-based method that incorporate with other features like entity position encoding or grammar relation.

<img width="461" alt="image" src="https://user-images.githubusercontent.com/84280247/220658280-0b80c51d-efe1-4036-a896-c77ebc7f8a73.png">

How does it run:
- process_data.py: gennerate X_train, X_test, y_train, y_test in numpy file from raw data in the relation-extraction/data, and create a data_encoder.obj
- model.py: use the data after being processed to train model, and create model saved in relation-extraction/saved_model
- predict.py: predict relation for a data sample, it use process_data.py to generate features, and data_encoder.obj to encode and then load the model saved to predict
- apps.py: demo app

I implement two approaches to represent the shortest dependency path feature to the model, the second approach is currently in the sp-bases branch

# Experiment and results
<img src="https://user-images.githubusercontent.com/84280247/223452855-087c57ef-70f0-4c17-9118-2169336332bf.png"  width="500">

# Demo
<img src="https://user-images.githubusercontent.com/84280247/223474150-4a567212-0ec5-4426-a766-daa9741c2ad7.png"  width="800">

