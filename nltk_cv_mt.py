from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import nltk_tfidf_mt_main as main
import numpy as np


###############################################################################
#------------------------ Count Vectorizer Function --------------------------#

def CountVect(aList):
    cv = CountVectorizer(max_features=1500, lowercase=False)
    X = cv.fit_transform(aList).toarray()
    # Sort, inverse and get indices of idf weights from more to less important
    X_sum = X.sum(axis=0)
    indices = np.argsort(X_sum)[::-1]
    # Get features
    feature_names = cv.get_feature_names()
    # Number of features
    top_n = 100
    # Save to top_features top_n features
    top_features = [feature_names[i] for i in indices[:top_n]]
    #cv.vocabulary_.items()]
    return(top_features, X)

################################## MAIN BODY ##################################
    
if __name__ == '__main__':
    
#---------------------- Call preprocessing Functions -------------------------#
# Load for preprocessing gr2en_dataset.txt that contains the machine translated
# tweets
    
    gr2en_translated_cv = main.LoadSavedGr2En()
    #print(pd.DataFrame(gr2en_translated_cv))
    tokenized_cv = main.Tokenize(gr2en_translated_cv)
    #print(pd.DataFrame(tokenized_cv))
    alpha_cv = main.NonAlphaLower(tokenized_cv)
    #print(pd.DataFrame(alpha_cv))
    stopwordsfree_cv = main.removeStopwords(alpha_cv)
    #print(pd.DataFrame(stopwordsfree_cv))
    lemmatized_cv = main.Lemmatizing(stopwordsfree_cv)
    #print(pd.DataFrame(lemmatized_cv))
    posTagged_cv = main.POSTagging(lemmatized_cv)
    #print(pd.DataFrame(posTagged_cv))

#-------------------- Call Feature Extraction Functions ----------------------#

    tweets_cv, labels_cv = main.DatasetSplit(posTagged_cv)
    #print(pd.DataFrame(tweets_cv))
    #print(pd.DataFrame(labels_cv))
    top_feat_cv, X_cv = CountVect(tweets_cv)
    tags_percent_cv = main.POSinTopFeat(top_feat_cv)
    #print(pd.DataFrame(tags_percent_cv))

#----------------------- Call TrainTestSet Function --------------------------#

    kfold_cv, X_train_cv, X_test_cv, y_train_cv, y_test_cv = \
    main.TrainTestSet(labels_cv, X_cv)
    
#------------------------ Call Classifier Functions ------------------------#

    RF_classifier_cv, RF_predictions_cv = \
    main.RFClassifier(X_train_cv, y_train_cv, X_test_cv)
    MNB_classifier_cv, MNB_predictions_cv = \
    main.MNBClassifier(X_train_cv, y_train_cv, X_test_cv)

#---------------------- Call Model Evaluation Function ---------------------#

    RF_conf_matrix_cv, RF_classif_report_cv, RF_accuracy_cv = \
    main.ModelEval(y_test_cv, RF_predictions_cv)
    #print(RF_conf_matrix_cv)
    #print(RF_classif_report_cv)
    #print(RF_accuracy_cv)
    MNB_conf_matrix_cv, MNB_classif_report_cv, MNB_accuracy_cv = \
    main.ModelEval(y_test_cv, MNB_predictions_cv)
    #print(MNB_conf_matrix_cv)
    #print(MNB_classif_report_cv)
    #print(MNB_accuracy_cv)

#------------------------ Call Compare Algos Function ------------------------#

    evaluation_cv, msg_cv = \
    main.CompareAlgos(RF_classifier_cv, MNB_classifier_cv, X_cv, labels_cv, \
                 kfold_cv)

###############################################################################