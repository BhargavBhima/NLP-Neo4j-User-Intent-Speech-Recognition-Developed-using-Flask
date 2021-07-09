#!flask/bin/python
from flask import Flask, jsonify
from flask import request,render_template

import pickle

# Tensorflow Model
import numpy as np
import tensorflow as tf
import tflearn
import random

from tensorflow.python.framework import ops

ops.reset_default_graph()

# UContextualisation and Other NLP Tasks.
import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

from keras import backend as K

import json
import pickle
import warnings

warnings.filterwarnings("ignore")

import nltk

nltk.download('punkt')

from py2neo import Graph

# model123 = Graph
graph = Graph(uri="bolt://localhost:7687", auth=("neo4j", "1234"))

app = Flask(__name__)



@app.route('/')
def index():
   return render_template('in.html')


@app.route('/result',methods = ['POST', 'GET'])
def result():

   if request.method == 'POST':
      #result = request.form
      input = request.form.get("Name", "")
      print("input is:", input)
      headers = {}
      # result = requests.get(f"http://127.0.0.1:5001/query?input={input}", headers= headers)
      result = query(input)
      print("result is: ", result)
      result = {"Name": result}


      return render_template("out.html",result = result)


def query(input):
   q = """
    MATCH (p) return (p) LIMIT 10
    """
   data = graph.run(q)
   l = []
   for d in data:
      l.append(d)
   print(l)

   q1 = """
    MATCH (p)-[:MAKES_PRODUCT]-(m:Product)-[:IMPORTED_FROM]-(c:Country)
    WHERE m.sector='healthcare' AND m.type="Masks" AND c.name="China"
    RETURN p
    """

   data1 = graph.run(q1)

   l1 = []
   for d1 in data1:
      l1.append(d1)

   q2 = "MATCH (p)-[:MAKES_PRODUCT]-(m)-[:IMPORTED_FROM]-(c:Country) WHERE m.sector='healthcare' RETURN p"

   data2 = graph.run(q2)
   l2 = []
   for d2 in data2:
      l2.append(d2)

   q3 = """
    MATCH (n)-[:MAKES_PRODUCT]-(m)-[:IMPORTED_FROM]-(c:Country)
    WHERE m.sector='healthcare' AND m.type="Masks"
    RETURN c.name, count(*) as high_dependence_country
    ORDER BY high_dependence_country DESC;
    """
   data3 = graph.run(q3)

   l3 = []
   for d3 in data3:
      l3.append(d3)

   intents = {"intents": [
      {"tag": "greeting",
       "patterns": ["Hi", "How are you", "Is anyone there?", "Hello", "Good day"],
       "responses": ["Hello, thanks for visiting", "Good to see you again", "Hi there, how can I help?"],
       "context_set": ""
       },
      {"tag": "goodbye",
       "patterns": ["Bye", "See you later", "Goodbye"],
       "responses": ["See you later, thanks for visiting", "Have a nice day", "Bye! Come back again soon."]
       },
      {"tag": "limit",
       "patterns": ["limit 10 nodes", "limit 10", "show 10", "only 10"],
       "responses": ["limit"]
       },
      {"tag": "healthcare",
       "patterns": ["list all the healthcare sector", "What organizations come under healthcare sector?",
                    "healthcare sector"],
       "responses": ["health"]
       },
      {"tag": "masks",
       "patterns": ["masks from china", "china masks"],
       "responses": ["masks"]
       },
      {"tag": "dependent",
       "patterns": ["most dependent countries on masks", "which countries are dependent on masks"],
       "responses": ["dependentmasks"]
       }]}

   words = []
   classes = []
   documents = []
   ignore_words = ['?']
   for intent in intents['intents']:
      for pattern in intent['patterns']:
         # tokenize each word in the sentence
         w = nltk.word_tokenize(pattern)
         # add to our words list
         words.extend(w)
         # add to documents in our corpus
         documents.append((w, intent['tag']))
         # add to our classes list
         if intent['tag'] not in classes:
            classes.append(intent['tag'])

   words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
   words = sorted(list(set(words)))

   # remove duplicates
   classes = sorted(list(set(classes)))

   training = []
   output = []
   output_empty = [0] * len(classes)

   for doc in documents:
      # initialize our bag of words
      bag = []
      # list of tokenized words for the pattern
      pattern_words = doc[0]
      # stem each word
      pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
      # create our bag of words array
      for w in words:
         bag.append(1) if w in pattern_words else bag.append(0)

      # output is a '0' for each tag and '1' for current tag
      output_row = list(output_empty)
      output_row[classes.index(doc[1])] = 1

      training.append([bag, output_row])

   random.shuffle(training)
   training = np.array(training)

   train_x = list(training[:, 0])
   train_y = list(training[:, 1])

   net = tflearn.input_data(shape=[None, len(train_x[0])])
   net = tflearn.fully_connected(net, 8)
   net = tflearn.fully_connected(net, 8)
   net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
   net = tflearn.regression(net)
   print("Training....")

   model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

   model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
   model.save('model.tflearn')

   pickle.dump({'words': words, 'classes': classes, 'train_x': train_x, 'train_y': train_y},
               open("training_data", "wb"))

   data = pickle.load(open("training_data", "rb"))
   words = data['words']
   classes = data['classes']
   train_x = data['train_x']
   train_y = data['train_y']

   with open('newintents.json') as json_data:
      intents = json.load(json_data)

   # load our saved model
   model.load('./model.tflearn')

   filename = 'model.pkl'
   pickle.dump(tflearn.DNN, open(filename, 'wb'))

   res = ""

   while True:

      answer = response(input, model, words, classes, intents)
      # answer
      if answer == "limit":
         print("answer value is - ", answer)
         res = l
         break
         print(*l, sep="\n")
      elif answer == "masks":
         res = l1
         print(*l1, sep="\n")
         break
      elif answer == "health":
         res = l2
         print(*l2, sep="\n")
         break
      elif answer == "dependentmasks":
         res = l3
         print(*l3, sep="\n")
         break
      else:
         res = "no nswer found"
         break

   return res


def clean_up_sentence(sentence):
   # It Tokenize or Break it into the constituents parts of Sentense.
   print("clean up sentence", sentence)
   sentence_words = nltk.word_tokenize(sentence)
   # Stemming means to find the root of the word.
   sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
   return sentence_words


# Return the Array of Bag of Words: True or False and 0 or 1 for each word of bag that exists in the Sentence
def bow(sentence, words, show_details=False):
   sentence_words = clean_up_sentence(sentence)
   bag = [0] * len(words)
   for s in sentence_words:
      for i, w in enumerate(words):
         if w == s:
            bag[i] = 1
            if show_details:
               print("found in bag: %s" % w)
   return (np.array(bag))


ERROR_THRESHOLD = 0.25


def classify(sentence, model, words, classes):
   # Prediction or To Get the Posibility or Probability from the Model
   results = model.predict([bow(sentence, words)])[0]
   # Exclude those results which are Below Threshold
   results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
   # Sorting is Done because heigher Confidence Answer comes first.
   results.sort(key=lambda x: x[1], reverse=True)
   return_list = []
   for r in results:
      return_list.append((classes[r[0]], r[1]))  # Tuppl -> Intent and Probability
   return return_list


def response(sentence, model, words, classes, intents, show_details=False):
   results = classify(sentence, model, words, classes)
   # That Means if Classification is Done then Find the Matching Tag.
   if results:
      # Long Loop to get the Result.
      while results:
         for i in intents['intents']:
            # Tag Finding
            if i['tag'] == results[0][0]:
               # Random Response from High Order Probabilities
               rand_response = random.choice(i['responses'])
               return str(rand_response)

         results.pop(0)


if __name__ == '__main__':
   app.run(debug = True)