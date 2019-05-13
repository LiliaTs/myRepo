import nltk
import xlrd
import pandas as pd
import csv
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from nltk.stem import WordNetLemmatizer
from google.cloud import translate
import pickle
import code_base_thesis_main as main
import random

###############################################################################

# READ reviews from files, do the LABELLING and save pos and neg reviews them
# in myList
# Read XLS reviews file
myFile = xlrd.open_workbook('sample_data_gr.xls')
mySheet = myFile.sheet_by_index(0)
num_rows = mySheet.nrows
tmpList = []
myList = []
for row in range(0, num_rows):
    myReviews = mySheet.cell_value(row, 1)
    myRatings = mySheet.cell_value(row, 4)
    tmpList = [myReviews, myRatings]
    myList.append(tmpList)

# Change ratings to labels
for review in myList:
    if review[1] == 1 or review[1] == 2:
        review[1] = 'Negative'
    elif review[1] == 4 or review[1] == 5:
        review[1] = 'Positive'        
#print(pd.DataFrame(myList))

# Remove neutral reviews
for review in myList:
    if review[1] == 3:
        myList.remove(review)
#print(pd.DataFrame(myList))

# Read CSV reviews file
with open('GRGE_small.csv', encoding="utf8") as myCSV:
    readCSV = csv.reader(myCSV, delimiter=',')
    tmpList = []
    for row in readCSV:
        #print(row)
        if row[2] != 'Neutral':
            tmpList = [row[1], row[2]]
            myList.append(tmpList)
#print(myList)
#print(len(myList))

###############################################################################

# # T-R-A-N-S-L-A-T-I-O-N FUNCTION - Google Cloud Translation API
translateClient = translate.Client()
def sentTranslation(aList):
    transList = []
    for review in aList:
        tmp = []
        translate = translateClient.translate(review[0], target_language='en')
        tmp = [translate['translatedText'], review[-1]]
        myStr = ' '.join(tmp)
        transList.append(myStr)
    return(transList)

# 1. Translation for whole SENTENCES (before tokenize)
# CALL preprocessing FUNCTIONS
cleanTweetsListSent = main.CleanTweets(myList)
# CALL translation FUNCTION
transSentList = sentTranslation(cleanTweetsListSent[:20])
# CALL the rest preprocessing FUNCTIONS
tokensListSent = main.Tokenize(transSentList)
alphaListSent = main.NonAlphaLower(tokensListSent)
stopwordsFreeListSent = main.removeStopwords(alphaListSent)
lemmatizedListSent = main.Lemmatizing(stopwordsFreeListSent)
taggedListSent = main.POSTagging(lemmatizedListSent)

# CALL feature extraction FUNCTIONS
allWordsSent = main.allWords(taggedListSent)
#print(all_words.plot(10)) #draw plot for first 10 words
myFeaturesSent = main.Features(allWordsSent)
myFeatureListSent = main.myReviewsFeatures(taggedListSent, myFeaturesSent)

# CLASSIFIERS
# a. Model based on this script reviews
# shuffle my features list or I can write some code to choose same # of pos &
# neg reviews for each set -> training set - 1000 (500 pos & 500 neg) reviews
random.shuffle(myFeatureListSent)

training_set = myFeatureListSent[:10]
testing_set = myFeatureListSent[10:]

# Naive Bayes
classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Naive Algo accuracy:", (nltk.classify.accuracy(classifier, testing_set))*100, '%')
classifier.show_most_informative_features(10)

#-------------------------------------VS--------------------------------------#

# b. TEST reviews based on the model from the main script 
# open myFeat file created in code_base.py and write to list
myFeatsList = []
with open('myFeatures.txt', 'r') as f:
    for line in f:
        myFeatures = line[:-1]
        myFeatsList.append(myFeatures)

# English features in translated-to-english greek reviews existence
myTransGrFeatureList = []
for review in taggedListSent:
    myFeatures = {}
    for word in myFeatsList:
        myFeatures[word] = (word in review[:-1])
        tmp = (myFeatures, review[-1])
    myTransGrFeatureList.append(tmp)

# Test using the saved model for english
# open classifier file from the main script
classifier_f = open("naivebayes.pickle", "rb")
classifierMain = pickle.load(classifier_f)
classifier_f.close()

print("Naive Algo accuracy percent:", (nltk.classify.accuracy(classifierMain, myTransGrFeatureList))*100)
classifierMain.show_most_informative_features(10)

# 1. translation for sentences (before tokenize) and then classifier
# 2. translation for words (after tokenize) and then classfier
# compare results and use code_base_main classifier for the best of the above