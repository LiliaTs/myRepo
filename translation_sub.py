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
import code_base_main as main
import random
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier


#---------------------------- Load Greek Dataset -----------------------------#
# Load greek dataset from file created at load_dataset_gr.py

def LoadReviews():
    greek_reviews = []
    with open('gr2en_files/greek_dataset.csv', 'r', encoding='utf-8', \
              newline='') as f:
        rd  = csv.reader(f)
        for line in rd:
            greek_reviews.append(line)
    return(greek_reviews)

#-------------------------- Translation Function -----------------------------#
# Using Google Cloud Translation API

def gr2enTranslation(aList):
    gr2en_translated = []
    for review in aList:
        tmp = []
        translate = translateClient.translate(review[0], target_language='en')
        tmp = [translate['translatedText'], review[-1]]
        myStr = ' '.join(tmp)
        gr2en_translated.append(myStr)
    return(gr2en_translated)
    
#---------------- Save gr2en_translated to gr2en_dataset.txt -----------------#

def Gr2En(aList):    
    with open('gr2en_files/gr2en_dataset.txt', 'w', encoding="utf-8") as f:
        for review in aList:
            f.write('%s\n' % review)

#------------------------- Load Translated Dataset ---------------------------#

def LoadSavedGr2En():
    greek_reviews = []
    with open('gr2en_files/gr2en_dataset.txt', 'r', encoding="utf-8") as f:
        for line in f:
            greek_reviews.append(line)
    return(greek_reviews)


################################## MAIN BODY ##################################
    
if __name__ == '__main__':
    
    translateClient = translate.Client() # Create a client 
    
#---------------------- Call preprocessing Functions -------------------------#

# FUNCTIONS USED TO LOAD, TRANSLATE & SAVE REVIEWS TO FILE gr2en_dataset.txt
# =============================================================================
#     greek_reviews = LoadReviews()
#     #print(pd.DataFrame(greek_reviews))
#     cleanTweets_gr = main.CleanTweets(greek_reviews)
#     #print(pd.DataFrame(cleanTweets_gr))
#     gr2en_translated = gr2enTranslation(cleanTweets_gr)
#     #print(pd.DataFrame(gr2en_translated))
#     Gr2En(gr2en_translated) # save to file
# =============================================================================


# Load for preprocessing gr2en_dataset.txt that contains the machine translated
# reviews
    
    gr2en_translated_saved = LoadSavedGr2En()
    #print(pd.DataFrame(gr2en_translated_saved))
    tokenized_gr = main.Tokenize(gr2en_translated_saved)
    #print(pd.DataFrame(tokenized)_gr)
    alpha_gr = main.NonAlphaLower(tokenized_gr)
    #print(pd.DataFrame(alpha_gr))
    stopwordsfree_gr = main.removeStopwords(alpha_gr)
    #print(pd.DataFrame(stopwordsfree_gr))
    lemmatized_gr = main.Lemmatizing(stopwordsfree_gr)
    #print(pd.DataFrame(lemmatized_gr))
    tagged_gr = main.POSTagging(lemmatized_gr)
    #print(pd.DataFrame(tagged_gr))

#-------------------- Call Feature Extraction Functions ----------------------#

    reviews_gr, labels_gr = main.DatasetSplit(tagged_gr)
    #print(pd.DataFrame(reviews_gr))
    #print(pd.DataFrame(labels_gr))
    top_feat_gr, X_gr = main.TfIdf(reviews_gr)
    #print(pd.DataFrame(top_feat_gr))
    tags_percent_gr = main.POSinTopFeat(top_feat_gr)
    #print(pd.DataFrame(tags_percent_gr))

#----------------------- Call TrainTestSet Function --------------------------#

    kfold_gr, X_train_gr, X_test_gr, y_train_gr, y_test_gr = \
    main.TrainTestSet(labels_gr, X_gr)
    
#------------------------ Call Classifier Functions ------------------------#

    RF_classifier_gr, RF_predictions_gr = \
    main.RFClassifier(X_train_gr, y_train_gr, X_test_gr)
    MNB_classifier_gr, MNB_predictions_gr = \
    main.MNBClassifier(X_train_gr, y_train_gr, X_test_gr)

#---------------------- Call Model Evaluation Function ---------------------#

    RF_conf_matrix_gr, RF_classif_report_gr, RF_accuracy_gr = \
    main.ModelEval(y_test_gr, RF_predictions_gr)
    #print(RFconf_matrix)
    #print(RFclassif_report)
    #print(RFaccuracy)
    MNB_conf_matrix, MNB_classif_report, MNB_accuracy = \
    main.ModelEval(y_test_gr, MNB_predictions_gr)
    #print(MNBconf_matrix)
    #print(MNBclassif_report)
    #print(MNBaccuracy)

#------------------------ Call Compare Algos Function ------------------------#

    evaluation_gr, msg_gr = \
    main.CompareAlgos(RF_classifier_gr, MNB_classifier_gr, X_gr, labels_gr, \
                      kfold_gr)
    
#------------------------- Call Save Model Function ------------------------#

    main.SaveModel('saved_models/RF_classifier_gr', RF_classifier_gr)
    main.SaveModel('saved_models/MNB_classifier_gr', MNB_classifier_gr)


###############################################################################