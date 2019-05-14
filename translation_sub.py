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


#---------------------------- Load Greek Dataset -----------------------------#

def LoadReviews():
    greek_reviews = []
    with open('greek_dataset.csv', 'r', encoding='utf-8', newline='') \
    as f:
        rd  = csv.reader(f)
        for line in rd:
            greek_reviews.append(line)
    return(greek_reviews)

#-------------------------- Translation Function -----------------------------#

# Using Google Cloud Translation API
translateClient = translate.Client()
def gr2enTranslation(aList):
    gr2en_translated = []
    for review in aList:
        tmp = []
        translate = translateClient.translate(review[0], target_language='en')
        tmp = [translate['translatedText'], review[-1]]
        myStr = ' '.join(tmp)
        gr2en_translated.append(myStr)
    return(gr2en_translated)
    
#-------------------- Save gr2en_translated to Gr2En.csv ---------------------#

def Gr2En():    
    with open('gr2en_dataset.txt', 'w', encoding="utf-8") as f:
        for review in gr2en_translated:
            f.write('%s\n' % review)

#------------------------- Load Translated Dataset ---------------------------#

def LoadSavedGr2En():
    greek_reviews = []
    with open('gr2en_dataset.txt', 'r', encoding="utf-8") as f:
        for line in f:
            greek_reviews.append(line)
    return(greek_reviews)

#gr2en_translated = LoadSavedGr2En()

################################## MAIN BODY ##################################
    
if __name__ == '__main__':
    
#---------------------- Call preprocessing Functions -------------------------#

    greek_reviews = LoadReviews()
    #print(pd.DataFrame(greek_reviews))
    gr_cleanTweets = main.CleanTweets(greek_reviews[:10])
    #print(pd.DataFrame(gr_cleanTweets))
    gr2en_translated = gr2enTranslation(gr_cleanTweets)
    #print(pd.DataFrame(gr2en_translated))
    Gr2En() # save to file
    
    
# =============================================================================
#     trans_tokenized = main.Tokenize(gr2en_translated)
#     #print(pd.DataFrame(trans_tokenized))
#     trans_alpha = main.NonAlphaLower(trans_tokenized)
#     #print(pd.DataFrame(trans_alpha))
#     trans_stopwordsfree = main.removeStopwords(trans_alpha)
#     #print(pd.DataFrame(trans_stopwordsfree))
#     trans_lemmatized = main.Lemmatizing(trans_stopwordsfree)
#     #print(pd.DataFrame(trans_lemmatized))
#     trans_tagged = main.POSTagging(trans_lemmatized)
#     #print(pd.DataFrame(trans_tagged))
# 
# #-------------------- Call Feature Extraction Functions ----------------------#
# 
#     trans_reviews, trans_labels = main.DatasetSplit(trans_tagged)
#     #print(pd.DataFrame(trans_reviews))
#     #print(pd.DataFrame(trans_labels))
#     trans_top_feat, trans_kfold, trans_X, trans_X_train, trans_X_test, \
#     trans_y_train, trans_y_test = main.TfIdf(trans_reviews, trans_labels)
#     #print(pd.DataFrame(trans_top_feat))
#     trans_tags_percent = main.POSinTopFeat()
#     #print(pd.DataFrame(trans_tags_percent))
# 
# 
# ############# Execute the following code when creating a new model ############
# #------------------------ Call Classifier Functions ------------------------#
# 
#     trans_RF_classifier, trans_RF_predictions = \
#     main.RFClassifier(trans_X_train, trans_y_train, trans_X_test)
#     trans_MNB_classifier, trans_MNB_predictions = \
#     main.MNBClassifier(trans_X_train, trans_y_train, trans_X_test)
# 
# #------------------------- Call Save Model Function ------------------------#
# 
#     main.SaveModel('trans_RF_classifier', trans_RF_classifier)
#     main.SaveModel('trans_MNB_classifier', trans_MNB_classifier)
#     
# #---------------------- Call Model Evaluation Function ---------------------#
# 
#     trans_RFconf_matrix, trans_RFclassif_report, trans_RFaccuracy = \
#     main.ModelEval(trans_y_test, trans_RF_predictions)
#     #print(trans_RFconf_matrix)
#     #print(trans_RFclassif_report)
#     #print(trans_RFaccuracy)
#     trans_MNBconf_matrix, trans_MNBclassif_report, trans_MNBaccuracy = \
#     main.ModelEval(trans_y_test, trans_MNB_predictions)
#     #print(trans_MNBconf_matrix)
#     #print(trans_MNBclassif_report)
#     #print(trans_MNBaccuracy)
# 
# ############## Execute the above code when creating a new model ###############
# 
# 
# ###############################################################################
# # Test machine-translated greek reviews using saved model for english reviews #
# #------------------------- Call Load Model Function --------------------------#
#     
#     trans_RF_classifier_sm, trans_RF_predictions_sm = \
#     main.LoadModel('RF_classifier')
#     trans_MNB_classifier_sm, trans_MNB_predictions_sm = \
#     main.LoadModel('MNB_classifier')
# 
#    
# #------------- Call Model Evaluation Function for Saved Models ---------------#
# 
#     trans_RFconf_matrix_sm, trans_RFclassif_report_sm, trans_RFaccuracy_sm = \
#     main.ModelEval(trans_y_test, trans_RF_predictions_sm)
#     #print(trans_RFconf_matrix_sm)
#     #print(trans_RFclassif_report_sm)
#     #print(trans_RFaccuracy_sm)
#     trans_MNBconf_matrix_sm, trans_MNBclassif_report_sm, trans_MNBaccuracy_sm\
#     = main.ModelEval(trans_y_test, trans_MNB_predictions_sm)   
#     #print(trans_MNBconf_matrix_sm)
#     #print(trans_MNBclassif_report_sm)
#     #print(trans_MNBaccuracy_sm)
#     
# #------------------------ Call Compare Algos Function ------------------------#
# 
#     trans_evaluation, trans_msg = main.CompareAlgos(trans_RF_classifier_sm, \
#                                    trans_MNB_classifier_sm)
# 
# ###############################################################################
# =============================================================================


