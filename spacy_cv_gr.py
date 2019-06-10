import pandas as pd
import nltk_tfidf_mt_main as main
import nltk_cv_mt as mt
import spacy_tfidf_gr as sp


################################ MAIN BODY ####################################

if __name__ == "__main__":

#---------------------- Call preprocessing Functions -------------------------#

    tweets_lbls_gr = sp.LoadTweets()
    #print(pd.DataFrame(tweets_lbls_gr))
    cleanTweets_gr = sp.CleanTweets(tweets_lbls_gr)
    #print(pd.DataFrame(cleanTweets_gr))
    to_string = sp.TweetsToStr(cleanTweets_gr)
    #print(pd.DataFrame(to_string))
    tokenized_gr = sp.Tokenize(to_string)
    #print(pd.DataFrame(tokenized_gr))
    lowerList_gr = sp.Lowercase(tokenized_gr)
    #print(pd.DataFrame(lowerList))
    stopwordsfree_gr = sp.RemoveStopWords(lowerList_gr)
    #print(pd.DataFrame(stopwordsfree_gr))
    toString = sp.TweetsToStr(stopwordsfree_gr)
    #print(pd.DataFrame(toString))
    lemmatized_gr = sp.Lemmatize(toString)
    #print(pd.DataFrame(lemmatized_gr))
    toStrings = sp.TweetsToStr(lemmatized_gr)
    #print(pd.DataFrame(toStrings))
    posTagged_gr = sp.PosTag(toStrings)
    #print(pd.DataFrame(posTagged_gr))
    
#-------------------- Call Feature Extraction Functions ----------------------#

    tweets_gr, labels_gr = main.DatasetSplit(posTagged_gr)
    #print(pd.DataFrame(tweets_gr))
    #print(pd.DataFrame(labels_gr))
    top_feat_gr, X_gr = mt.CountVect(tweets_gr)
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