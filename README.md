# Relation-extraction
My first project in NLP

How it run:
- process_data.py: gennerate X_train, X_test, y_train, y_test in numpy file from raw data in the relation-extraction/data, and create a data_encoder.obj
- model.py: use the data after being processed to train model, and create model saved in relation-extraction/saved_model
- predict.py: predict relation for a data sample, it use process_data.py to generate features, and data_encoder.obj to encode and then load the model saved to predict
- apps.py: demo app
