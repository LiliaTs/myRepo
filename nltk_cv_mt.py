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
    
    gr2en_translated_mt_cv = main.LoadSavedGr2En()
    tokenized_mt_cv = main.Tokenize(gr2en_translated_mt_cv)
    alpha_mt_cv = main.NonAlphaLower(tokenized_mt_cv)
    stopwordsfree_mt_cv = main.removeStopwords(alpha_mt_cv)
    lemmatized_mt_cv = main.Lemmatizing(stopwordsfree_mt_cv)
    posTagged_mt_cv = main.POSTagging(lemmatized_mt_cv)

#-------------------- Call Feature Extraction Functions ----------------------#

    tweets_mt_cv, labels_mt_cv = main.DatasetSplit(posTagged_mt_cv)
    top_feat_mt_cv, X_mt_cv = CountVect(tweets_mt_cv)
    tags_pct_mt_cv = main.POSinTopFeat(top_feat_mt_cv)

#----------------------- Call TrainTestSet Function --------------------------#

    kfold_mt_cv, X_train_mt_cv, X_test_mt_cv, y_train_mt_cv, y_test_mt_cv = \
    main.TrainTestSet(labels_mt_cv, X_mt_cv)
    
#------------------------ Call Classifier Functions ------------------------#

    RF_clf_mt_cv, RF_pred_mt_cv = \
    main.RFClassifier(X_train_mt_cv, y_train_mt_cv, X_test_mt_cv)
    MNB_clf_mt_cv, MNB_pred_mt_cv = \
    main.MNBClassifier(X_train_mt_cv, y_train_mt_cv, X_test_mt_cv)

#---------------------- Call Model Evaluation Function ---------------------#

    RF_confm_mt_cv, RF_cls_rep_mt_cv, RF_acc_mt_cv = \
    main.ModelEval(y_test_mt_cv, RF_pred_mt_cv)
    MNB_confm_mt_cv, MNB_cls_rep_cv, MNB_acc_mt_cv = \
    main.ModelEval(y_test_mt_cv, MNB_pred_mt_cv)

#------------------------ Call Compare Algos Function ------------------------#

    all_acc_mt_cv, mean_acc_dev_mt_cv = \
    main.CompareAlgos(RF_clf_mt_cv, MNB_clf_mt_cv, X_mt_cv, labels_mt_cv, \
                 kfold_mt_cv)

###############################################################################