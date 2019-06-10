import spacy
import csv
import re
import pandas as pd
import nltk_tfidf_mt_main as main

########################### PREPROCESSING FUNCTIONS ###########################
#--------------- Remove @users, RTs, #hashtags, urls Function ----------------#

def CleanTweets(aList):
    cleanTweetsList = []
    for tweet in aList:
        myStr = ' '. join(tweet[:-1])
        removeUser = re.sub(r"@\S*", "", myStr)
        removeRTs = re.sub(r"RT\S*", "", removeUser)
        removeHTT = re.sub(r"htt\S*", "", removeRTs)
        removeURL = re.sub(r"\S*\.gr|\S*\.com", "", removeHTT)
        removeHashtag = re.sub(r"#\S*", "", removeURL)
        removePunctuation = re.sub(r"[^\w\s]", "", removeHashtag)
        removeDigits = re.sub(r"\d", "", removePunctuation)
        removeSpaces = re.sub(r" +", " ", removeDigits)
        removeSpace = re.sub(r"^ *", "", removeSpaces)
        cleanTweetsList.append([removeSpace, tweet[-1]])
    return(cleanTweetsList)

#----------------------------- TOKENIZE Function -----------------------------#

def Tokenize(aList):
    tokensList = []
    nlp = spacy.load('el_core_news_sm')
    for tweet in aList:
        rev = nlp(tweet)
        tmp  = [word.text for word in rev[:-1]]
        tokensList.append(tmp + [str(rev[-1])])
    return(tokensList)

#------------ LOWERCASE Function -------------#

# Lowercase and remove words with length <= 2
def Lowercase(aList):
    lowerList = []
    for tweet in aList:
        tmp = []
        for word in tweet[:-1]:
            #removeGreek = re.sub(r"[A-Za-z]+", "", word)
            if len(word)>2:
                tmp.append(word.lower())
        lowerList.append(tmp + [tweet[-1]])
    return(lowerList)

#------------- Remove STOP WORDS Function - works for lowercase! -------------#

def RemoveStopWords(aList):
    StopWords = spacy.lang.el.stop_words.STOP_WORDS
    stopWordsFree = []
    for tweet in aList:
        tmp = []
        for word in tweet[:-1]:
            if word not in StopWords:
                tmp.append(word)
        stopWordsFree.append(tmp + [tweet[-1]])
    return(stopWordsFree)

#--------------------------- LEMMATIZING Function ----------------------------#

def Lemmatize(aList):
    lemmaList = []
    nlp = spacy.load('el_core_news_sm')
    for tweet in aList:
        rev = nlp(tweet)
        tmp = [word.lemma_ for word in rev[:-1]]
        lemmaList.append(tmp + [str(rev[-1])])
    return(lemmaList)

#------------------- POS - Part of speech tagging Function -------------------#

def PosTag(aList):
    posList = []
    nlp = spacy.load('el_core_news_sm')
    for tweet in aList:
        rev = nlp(tweet)
        tmp = [(word.text, word.pos_) for word in rev[:-1]]
        #merge word and pos in a single string
        temp = [(word[0] + '_' + word[1]) for word in tmp]
        posList.append(temp + [str(rev[-1])])
    return(posList)
    
#---------------------------- Load Greek Dataset -----------------------------#
# Load greek dataset from file created at load_dataset_gr.py

def LoadTweets():
    greek_tweets = []
    with open('greek_files/greek_dataset.csv', 'r', encoding='utf-8', \
              newline='') as f:
        rd  = csv.reader(f)
        for line in rd:
            greek_tweets.append(line)
    return(greek_tweets)

def TweetsToStr(aList):
# Convert tweets to string to perform tokenize
    myStringsList = []
    for tweet in aList:
        myString = ' '.join(tweet)
        myStringsList.append(myString)
    return(myStringsList)


################################ MAIN BODY ####################################

if __name__ == "__main__":

#---------------------- Call preprocessing Functions -------------------------#

    tweets_lbls_gr = LoadTweets()
    #print(pd.DataFrame(tweets_lbls_gr))
    cleanTweets_gr = CleanTweets(tweets_lbls_gr)
    #print(pd.DataFrame(cleanTweets_gr))
    to_string = TweetsToStr(cleanTweets_gr)
    #print(pd.DataFrame(to_string))
    tokenized_gr = Tokenize(to_string)
    #print(pd.DataFrame(tokenized_gr))
    lowerList_gr = Lowercase(tokenized_gr)
    #print(pd.DataFrame(lowerList))
    stopwordsfree_gr = RemoveStopWords(lowerList_gr)
    #print(pd.DataFrame(stopwordsfree_gr))
    toString = TweetsToStr(stopwordsfree_gr)
    #print(pd.DataFrame(toString))
    lemmatized_gr = Lemmatize(toString)
    #print(pd.DataFrame(lemmatized_gr))
    toStrings = TweetsToStr(lemmatized_gr)
    #print(pd.DataFrame(toStrings))
    posTagged_gr = PosTag(toStrings)
    #print(pd.DataFrame(posTagged_gr))
    
#-------------------- Call Feature Extraction Functions ----------------------#

    tweets_gr, labels_gr = main.DatasetSplit(posTagged_gr)
    #print(pd.DataFrame(tweets_gr))
    #print(pd.DataFrame(labels_gr))
    top_feat_gr, X_gr = main.TfIdf(tweets_gr)
    #print(pd.DataFrame(top_feat_gr))
    tags_percent_gr = main.POSinTopFeat(top_feat_gr)
    #print(pd.DataFrame(tags_percent_gr))

#----------------------- Call TrainTestSet Function --------------------------#

    kfold_gr, X_train_gr, X_test_gr, y_train_gr, y_test_gr = \
    main.TrainTestSet(labels_gr, X_gr)
    
#------------------------ Call Classifier Functions --------------------------#

    RF_classifier_gr, RF_predictions_gr = \
    main.RFClassifier(X_train_gr, y_train_gr, X_test_gr)
    MNB_classifier_gr, MNB_predictions_gr = \
    main.MNBClassifier(X_train_gr, y_train_gr, X_test_gr)

#---------------------- Call Model Evaluation Function -----------------------#

    RF_conf_matrix_gr, RF_classif_report_gr, RF_accuracy_gr = \
    main.ModelEval(y_test_gr, RF_predictions_gr)
    #print(RF_conf_matrix_gr)
    #print(RF_classif_report_gr)
    #print(RF_accuracy_gr)
    MNB_conf_matrix_gr, MNB_classif_report_gr, MNB_accuracy_gr = \
    main.ModelEval(y_test_gr, MNB_predictions_gr)
    #print(MNB_conf_matrix_gr)
    #print(MNB_classif_report_gr)
    #print(MNB_accuracy_gr)

#------------------------ Call Compare Algos Function ------------------------#

    all_accuracies_gr, mean_accur_dev_gr = \
    main.CompareAlgos(RF_classifier_gr, MNB_classifier_gr, X_gr, labels_gr, \
                      kfold_gr)
    
#------------------------- Call Save Model Function --------------------------#

    main.SaveModel('saved_models/RF_classifier', RF_classifier_gr)
    main.SaveModel('saved_models/MNB_classifier', MNB_classifier_gr)