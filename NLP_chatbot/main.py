import random
import json
import pickle
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import nltk
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
intent = json.loads(open(r'NLP_chatbot/intents.json').read())

words = []
classes = []
documents = []
ignoreLetters = ["?","!",".",","]

for intent in intent['intents']:
    for pattern in intent['patterns']:
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList) # add the tokens from current pattern
        documents.append((wordList, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters]
words = sorted(set(words))

classes = sorted(set(classes))

pickle.dump(words, open('NLP_chatbot/words.pkl', 'wb'))
pickle.dump(classes, open('NLP_chatbot/classes.pkl', 'wb'))

training =[]
outputEmpty =[0] * len(classes)

for document in documents:
    bag =[]
    wordPatterns = document[0]
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
    for word in words:
        bag.append(1) if word in wordPatterns else bag.append(0)

    
    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1
    training.append(bag + outputRow)


random.shuffle(training)
training = np.array(training)

trainX = training[:,: len(words)]
trainY = training[:, len(words):]

# creating model

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128,input_shape = (len(trainX[0]),),activation = "relu"))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64,activation = "relu"))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(trainY[0]), activation = "softmax"))

sgd = tf.keras.optimizers.SGD(learning_rate = 0.01, momentum = 0.9, nesterov = True)

model.compile(loss = "categorical_crossentropy", optimizer = sgd, metrics = ["accuracy"])

hist = model.fit(
    np.array(trainX),np.array(trainY), epochs= 250, batch_size = 5, verbose = 1
)

model.save('NLP_chatbot/chatbotModel.h5')
print("----DONE----")