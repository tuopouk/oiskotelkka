# oiskotelkka

By using Xeno-canto.org API and bird sound recordings (https://www.xeno-canto.org), I fitted a classification model to recognize bird species from their singing. I use RandomForestClassifier with Mel Frequency Cepstrums as features.

One can upload a .wav file to test the service. With only Finnish recording as training data, I was only able to perfrom a 16% accuracy which is why the name of the service "Oisko Telkka?" (in English "I wonder if it is a Common Goldeneye" referring to the uncertainty of the model). I was able to build the model by using KNN. I'll keep working on the accuracy. I also updated the model by using all Finnish samples to train it.

I store the model, label encoder and scaler in pickled documents so this uses a prefitted model that should be fast to apply for the user.

