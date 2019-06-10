import pandas as pd
import nltk_tfidf_mt_main as main
import nltk_cv_mt as mt
import spacy_tfidf_gr as sp


################################ MAIN BODY ####################################

if __name__ == "__main__":

#---------------------- Call preprocessing Functions -------------------------#

    tweets_lbls_sp_gr_cv = sp.LoadTweets()
    cleanTweets_sp_gr_cv = sp.CleanTweets(tweets_lbls_sp_gr_cv)
    to_string_sp_gr_cv = sp.TweetsToStr(cleanTweets_sp_gr_cv)
    tokenized_sp_gr_cv = sp.Tokenize(to_string_sp_gr_cv)
    lowerList_sp_gr_cv = sp.Lowercase(tokenized_sp_gr_cv)
    stopwordsfree_sp_gr_cv = sp.RemoveStopWords(lowerList_sp_gr_cv)
    toString_sp_gr_cv = sp.TweetsToStr(stopwordsfree_sp_gr_cv)
    lemmatized_sp_gr_cv = sp.Lemmatize(toString_sp_gr_cv)
    toStrings_sp_gr_cv = sp.TweetsToStr(lemmatized_sp_gr_cv)
    posTagged_sp_gr_cv = sp.PosTag(toStrings_sp_gr_cv)
    
#-------------------- Call Feature Extraction Functions ----------------------#

    tweets_sp_gr_cv, labels_sp_gr_cv = main.DatasetSplit(posTagged_sp_gr_cv)
    top_feat_sp_gr_cv, X_sp_gr_cv = mt.CountVect(tweets_sp_gr_cv)
    tags_pct_sp_gr_cv = main.POSinTopFeat(top_feat_sp_gr_cv)

#----------------------- Call TrainTestSet Function --------------------------#

    kfold_sp_gr_cv, X_train_sp_gr_cv, X_test_sp_gr_cv, y_train_sp_gr_cv, \
    y_test_sp_gr_cv = main.TrainTestSet(labels_sp_gr_cv, X_sp_gr_cv)
    
#------------------------ Call Classifier Functions --------------------------#

    RF_clf_sp_gr_cv, RF_pred_sp_gr_cv = \
    main.RFClassifier(X_train_sp_gr_cv, y_train_sp_gr_cv, X_test_sp_gr_cv)
    MNB_clf_sp_gr_cv, MNB_pred_sp_gr_cv = \
    main.MNBClassifier(X_train_sp_gr_cv, y_train_sp_gr_cv, X_test_sp_gr_cv)

#---------------------- Call Model Evaluation Function -----------------------#

    RF_confm_sp_gr_cv, RF_cls_rep_sp_gr_cv, RF_acc_sp_gr_cv = \
    main.ModelEval(y_test_sp_gr_cv, RF_pred_sp_gr_cv)
    MNB_conm_sp_gr_cv, MNB_cls_rep_sp_gr_cv, MNB_acc_sp_gr_cv = \
    main.ModelEval(y_test_sp_gr_cv, MNB_pred_sp_gr_cv)

#------------------------ Call Compare Algos Function ------------------------#

    all_acc_sp_gr_cv, mean_acc_dev_sp_gr_cv = \
    main.CompareAlgos(RF_clf_sp_gr_cv, MNB_clf_sp_gr_cv, X_sp_gr_cv, \
                      labels_sp_gr_cv, kfold_sp_gr_cv)
    
#------------------------- Call Save Model Function --------------------------#

    main.SaveModel('saved_models/RF_classifier', RF_clf_sp_gr_cv)
    main.SaveModel('saved_models/MNB_classifier', MNB_clf_sp_gr_cv)