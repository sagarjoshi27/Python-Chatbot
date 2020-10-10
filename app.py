#!/usr/bin/env python3
"""
Sagar Joshi N0774756 - Chatbot
"""

#  Initialise different libraries
import aiml
from flask import Flask 
from flask import render_template
import os
import nltk
import numpy as np
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image 
from seq2seq import *
import sys 


#Initialising FOL Logic Agent
v="""
nikes => {}
adidas => {}
pumas => {}
jordans => {}
reeboks => {}
vans => {}
offwhites => {}
jimmychoos => {}
valentinos => {}
balenciagas => {}
guccis => {}
topmans => {}
tommyhilfigers => {}
aldos => {}
clarks => {}
drmartens => {}
trainers => t1
boots => t2
smarts => t3
designers=> t4
be_in=>{}
"""
folval=nltk.Valuation.fromstring(v)
grammar_file='simple-sem.fcfg'
objectCounter=0

#Initialise AIML File
kernel = aiml.Kernel()
kernel.learn("mybot-basic.xml")

###########################################################

#Open's Question and Answer Text file
file = open("mybot.txt", "r")
questionList = []
answerList = []

for line in file:
    fields = line.split("$")  
    questionList.append(fields[0].lower())
    answerList.append(fields[1])

#Initiates WordNetLemmatizer from NLTK Library
wnlemmatizer = nltk.stem.WordNetLemmatizer()

#Lemmatizes list of words
def perform_lemmatization(tokens):
    return [wnlemmatizer.lemmatize(token) for token in tokens]

#Removes punctuation from text
punctuation_removal = dict((ord(punctuation), None) for punctuation in string.punctuation)

#Tokenizes, Lemmatizes and Removes Punctuation From The Sentence
def get_processed_text(document):
    return perform_lemmatization(nltk.word_tokenize(document.lower().translate(punctuation_removal)))


#######################################################################################
    
# Main loop
def findAnswer(userinput):
    #pre-process user input and determine response agent (if needed)
    responseAgent = 'query'
    global objectCounter

    #activate selected response agent
    if responseAgent == 'query':
        answer = kernel.respond(userinput)
    #post-process the answer for commands
    if answer[0] == '#':
        params = answer[1:].split('$')
        cmd = int(params[0])
        if cmd == 0:
            return(params[1])
            sys.exit()
        elif cmd == 1:
            #Loads and uses trained neural network
            model= tf.keras.models.load_model("sport_identifier.h5")
            
            userPath = userinput[8:]          
            imagePath = userPath
            imagePrediction = image.load_img(imagePath,target_size = (60,60))
            imagePrediction = image.img_to_array(imagePrediction)
            imagePrediction = np.expand_dims(imagePrediction, axis = 0)
            
            output = model.predict(imagePrediction)
            if output[0][1] == 1:
                prediction = "baseball"
            elif output[0][2] == 1:
                prediction = "golf"
            elif output[0][3] == 1:
                prediction = "basketball"
            elif output[0][4] == 1:
                prediction = "soccor"
            else:
                prediction = "unknown"
            return "This image indicates the sport is " + prediction
        elif cmd == 99:
            botlux_response = '' 
            questionList.append(userinput) 

            #Initiates tfidvectorizer which vectorises the questions
            word_vectorizer = TfidfVectorizer(tokenizer=get_processed_text, stop_words='english')
            all_word_vectors = word_vectorizer.fit_transform(questionList)

            #Finds cosine similarity and word vectors
            similar_vector_values = cosine_similarity(all_word_vectors[-1], all_word_vectors)
            similar_sentence_number = similar_vector_values.argsort()[0][-2]
            
            #Flattens the retrieved cosine similarity
            matched_vector = similar_vector_values.flatten()
            matched_vector.sort()
            vector_matched = matched_vector[-2]

            #If vector = 0 then bot hasn't found an answer
            if vector_matched == 0:
                botlux_response = botlux_response + "I am sorry, I could not understand you"
                return botlux_response
            else:
                botlux_response = botlux_response + answerList[similar_sentence_number]
                return botlux_response
        elif cmd == 4: # I will put x in y category
            o = 'o' + str(objectCounter)
            objectCounter += 1
            folval['o' + o] = o #insert constant
            if len(folval[params[1]]) == 1: #clean up if necessary
                if ('',) in folval[params[1]]:
                    folval[params[1]].clear()
            folval[params[1]].add((o,)) #insert type of shoe information
            if len(folval["be_in"]) == 1: #clean up if necessary
                if ('',) in folval["be_in"]:
                    folval["be_in"].clear()
            folval["be_in"].add((o, folval[params[2]])) #insert location
        elif cmd == 5: #Are there any x in category y
             fol_reponse = ''
             g = nltk.Assignment(folval.domain)
             m = nltk.Model(folval.domain, folval)
             sent = 'some ' + params[1] + ' are_in ' + params[2]
             results = nltk.evaluate_sents([sent], grammar_file, m, g)[0][0]
             if results[2] == True:
                 fol_reponse = fol_reponse + "Yes."
                 return fol_reponse
             else:
                 fol_reponse = fol_reponse + "No."
                 return fol_reponse
        elif cmd == 6: # Are all x in category y
             fol_reponse = ''
             g = nltk.Assignment(folval.domain)
             m = nltk.Model(folval.domain, folval)
             sent = 'all ' + params[1] + ' are_in ' + params[2]
             results = nltk.evaluate_sents([sent], grammar_file, m, g)[0][0]
             if results[2] == True:
                fol_reponse = fol_reponse + "Yes."
                return fol_reponse
                #print("Yes.")
             else:
                 fol_reponse = fol_reponse + "No."
                 return fol_reponse
        elif cmd == 7: # Are all in category y .....
            fol_reponse = ''
            g = nltk.Assignment(folval.domain)
            m = nltk.Model(folval.domain, folval)
            e = nltk.Expression.fromstring("be_in(x," + params[1] + ")")
            sat = m.satisfiers(e, "x", g)
            if len(sat) == 0:
                fol_reponse = fol_reponse + "None."
                return fol_reponse
            else:
                #find satisfying objects in the valuation dictionary,
                #and print their type names
                sol = folval.values()
                for so in sat:
                    for k, v in folval.items():
                        if len(v) > 0:
                            vl = list(v)
                            if len(vl[0]) == 1:
                                for i in vl:
                                    if i[0] == so:
                                        fol_reponse = fol_reponse + k
                                        return fol_reponse
                                        break
        elif cmd == 8:
            model = Model(inputs=[encoder_input, decoder_input], outputs=[decoder_outtput])
            adam = Adam() #Adam(lr=0.1, decay=0.0005)
            model.load_weights("model.h5")
            model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
            prediction = model.predict(userinput)
            return prediction
    else:
        return(answer)
    
###################################################################################

#Initialise Web UI
app = Flask(__name__)

@app.route("/",endpoint = 'index1')
def index1():
    return render_template("index.html")

@app.route("/<query>", endpoint = 'index2') #You have to return the answer here
def index2(query):
    return findAnswer(query)

if __name__ == "__main__":
    app.run
