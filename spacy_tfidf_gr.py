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

    tweets_lbls_sp_gr_idf = LoadTweets()
    cleanTweets_sp_gr_idf = CleanTweets(tweets_lbls_sp_gr_idf)
    to_string_sp_gr_idf = TweetsToStr(cleanTweets_sp_gr_idf)
    tokenized_sp_gr_idf = Tokenize(to_string_sp_gr_idf)
    lowerList_sp_gr_idf = Lowercase(tokenized_sp_gr_idf)
    stopwordsfree_sp_gr_idf = RemoveStopWords(lowerList_sp_gr_idf)
    toString_sp_gr_idf = TweetsToStr(stopwordsfree_sp_gr_idf)
    lemmatized_sp_gr_idf = Lemmatize(toString_sp_gr_idf)
    toStrings_sp_gr_idf = TweetsToStr(lemmatized_sp_gr_idf)
    posTagged_sp_gr_idf = PosTag(toStrings_sp_gr_idf)
    
#-------------------- Call Feature Extraction Functions ----------------------#

    tweets_sp_gr_idf, labels_sp_gr_idf = main.DatasetSplit(posTagged_sp_gr_idf)
    top_feat_sp_gr_idf, X_sp_gr_idf = main.TfIdf(tweets_sp_gr_idf)
    tags_pct_sp_gr_idf = main.POSinTopFeat(top_feat_sp_gr_idf)

#----------------------- Call TrainTestSet Function --------------------------#

    kfold_sp_gr_idf, X_train_sp_gr_idf, X_test_sp_gr_idf, y_train_sp_gr_idf, \
    y_test_sp_gr_idf = main.TrainTestSet(labels_sp_gr_idf, X_sp_gr_idf)
    
#------------------------ Call Classifier Functions --------------------------#

    RF_clf_sp_gr_idf, RF_pred_sp_gr_idf = \
    main.RFClassifier(X_train_sp_gr_idf, y_train_sp_gr_idf, X_test_sp_gr_idf)
    MNB_clf_sp_gr_idf, MNB_pred_sp_gr_idf = \
    main.MNBClassifier(X_train_sp_gr_idf, y_train_sp_gr_idf, X_test_sp_gr_idf)

#---------------------- Call Model Evaluation Function -----------------------#

    RF_conf_matrix_gr, RF_classif_report_gr, RF_accuracy_gr = \
    main.ModelEval(y_test_sp_gr_idf, RF_pred_sp_gr_idf)
    MNB_confm_sp_gr_idf, MNB_cls_rep_sp_gr_idf, MNB_acc_sp_gr_idf = \
    main.ModelEval(y_test_sp_gr_idf, MNB_pred_sp_gr_idf)

#------------------------ Call Compare Algos Function ------------------------#

    all_acc_sp_gr_idf, mean_acc_sp_gr_idf = \
    main.CompareAlgos(RF_clf_sp_gr_idf, MNB_clf_sp_gr_idf, X_sp_gr_idf, \
                      labels_sp_gr_idf, kfold_sp_gr_idf)
    
#------------------------- Call Save Model Function --------------------------#

    main.SaveModel('saved_models/RF_classifier', RF_clf_sp_gr_idf)
    main.SaveModel('saved_models/MNB_classifier', MNB_clf_sp_gr_idf)