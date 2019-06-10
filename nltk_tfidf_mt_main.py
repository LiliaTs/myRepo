import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, \
accuracy_score
import pandas as pd
import numpy as np
import re
import pickle
from collections import Counter
import matplotlib.pyplot as plt


########################### PREPROCESSING FUNCTIONS ###########################
#----------------------------- TOKENIZE Function -----------------------------#

def Tokenize(aList):
    tokensList = []
    for tweet in aList:
        tokens = word_tokenize(tweet)
        tokensList.append(tokens)
    return(tokensList)

#----- Remove NON-ALPHA characters, words len <=2 and LOWERCASE Function -----#

def NonAlphaLower(aList):
    alphaList = []
    for tweet in aList:
        tmpList = []
        for word in tweet[:-1]:
            removeGreek = re.sub(r"[^A-Za-z]+", "", word)
            if len(removeGreek) > 2 and removeGreek.isalpha():
                tmpList.append(removeGreek.lower())
        alphaList.append(tmpList + [tweet[-1]])
    return(alphaList)

#------------- Remove STOP WORDS Function - works for lowercase! -------------#

def removeStopwords(aList):
    myStopwords = set(stopwords.words('english'))
    stopwordsFreeList = []
    for tweet in aList:
        tmpList = []
        for word in tweet[:-1]:
            if word not in myStopwords:
                tmpList.append(word)
        stopwordsFreeList.append(tmpList + [tweet[-1]])
    return(stopwordsFreeList)

#--------------------------- LEMMATIZING Function ----------------------------#

def Lemmatizing(aList):
    lm = WordNetLemmatizer()
    lemmatizedList = []
    for tweet in aList:
            tmpList = []
            for word in tweet[:-1]:
                tmpList.append(lm.lemmatize(word))
            lemmatizedList.append(tmpList + [tweet[-1]])
    return(lemmatizedList)

#------------------- POS - Part of speech tagging Function -------------------#

def POSTagging(aList):
    taggedList = []
    for tweet in aList:
        tagged = nltk.pos_tag(tweet[:-1])
        new_tagged = []
        for word in tagged:
            new_tagged.append(word[0] + '_' + word[1]) #merge word and tag
        taggedList.append(new_tagged + [tweet[-1]])
    return(taggedList)


######################## FEATURE EXTRACTION FUNCTIONS #########################
#-------------------------- DatasetSplit Function ----------------------------#

def DatasetSplit(aList):
    # Prepare dataset for feature extraction - Split tweets from labels
    # Split taggedList to tweetsList and labelsList
    tweetsList = []
    labelsList = []
    for tweet in aList:
        tweetsList.append(', '.join(tweet[:-1]))
        labelsList.append(tweet[-1])
    # Convert labels to 1s and 0s
    labelsList[:] = [1 if (label == 'Positive') else 0 for label in labelsList]
    return(tweetsList, labelsList)

#----------------------------- TF-IDF Function -------------------------------#

def TfIdf(aList):
#    tf = TfidfVectorizer(max_features=1500, lowercase=False, min_df=4, \
#                         max_df=0.7)
    #tf = TfidfVectorizer(max_features=1500, lowercase=False)
    tf = TfidfVectorizer(lowercase=False)
    X = tf.fit_transform(aList).toarray()
    # Sort, inverse and get indices of idf weights from more to less important
    indices = np.argsort(tf.idf_)[::-1]
    # Get features
    feature_names = tf.get_feature_names()
    # Number of features
    top_n = 100
    # Save to top_features top_n features
    top_features = [feature_names[i] for i in indices[:top_n]]
    return(top_features, X)

#-------------------------- TrainTestSet Function ----------------------------#

def TrainTestSet(aList, X):
    Lbls = np.array(aList)
    kfold = KFold(n_splits=5, shuffle=True, random_state=1)
    for train_set,test_set in kfold.split(X):
        X_train, X_test = X[train_set], X[test_set]
        y_train, y_test = Lbls[train_set], Lbls[test_set]
    
    return(kfold, X_train, X_test, y_train, y_test)

#--------------------------- POSinTopFeat Function ---------------------------#
#% of POS tags participating in 100 top features Function

def POSinTopFeat(top_features):
    word_tags = []
    for f in top_features:
        tmp = f.split('_')
        word_tags.append(tmp)
    tags = []    
    # Count POS in word_tags and sort based on most common
    for i in word_tags:
        tags.append(i[1])
    cnt = Counter(tags)
    counted_tags = cnt.most_common()
    #print(counted_tags)    
    # Calculate percentage of POS
    result = []
    for tag,counted in counted_tags:
        tmp = str(tag) + ': ' + str(counted) + ' %'
        result.append(tmp)
    return(result)
    

########################### CLASSIFIERS FUNCTIONS #############################
#--------------------- Random Forest Classifier Function ---------------------#

def RFClassifier(X_train, y_train, X_test):
    RF_classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
    # Train the algorithm
    RF_classifier.fit(X_train, y_train)
    # Predict sentiment
    RF_predictions = RF_classifier.predict(X_test)
    return(RF_classifier, RF_predictions)
    
#---------------- Multinomial Naive Bayes Classifier Function ----------------#

def MNBClassifier(X_train, y_train, X_test):
    MNB_classifier = MultinomialNB(alpha=0.5)
    # Train the algorithm
    MNB_classifier.fit(X_train, y_train)
    # Predict sentiment
    MNB_predictions = MNB_classifier.predict(X_test)
    return(MNB_classifier, MNB_predictions)

#------------------------ Model Evaluation - Metrics -------------------------#

def ModelEval(y_test, RF_predictions):
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, RF_predictions)
    # Classification report
    classif_report = classification_report(y_test, RF_predictions)
    # Accuracy score
    accuracy = accuracy_score(y_test, RF_predictions)*100
    return(conf_matrix, classif_report, accuracy)

#--------------------- Compare RF - MNB algos Function -----------------------#

def CompareAlgos(RF_classifier, MNB_classifier, X, labelsList, kfold):
    models = []
    results = []
    names = []
    mean_acc_dev = []
    all_acc = []
    models.append(('RF', RF_classifier))
    models.append(('MNB', MNB_classifier))
    for name, model in models:
        cv_evaluation = cross_val_score(model, X, labelsList, cv=kfold)
        results.append(cv_evaluation)
        names.append(name)
        message = (name, cv_evaluation.mean(), cv_evaluation.std())
        mean_acc_dev.append(message)
        all_acc.append(cv_evaluation)
    # boxplot algorithm comparison
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()
    return(all_acc, mean_acc_dev)

#---------------------------- SaveModel Function -----------------------------#

def SaveModel(filePath, classifier):
    # Save model to filePath pickle file for direct use
    with open(filePath, 'wb') as picklefile:
        pickle.dump(classifier, picklefile)
        
#---------------------------- LoadModel Function -----------------------------#

def LoadModel(filePath):
    # Load model from filePath and stored_model variable
    with open(filePath, 'rb') as training_model:
        stored_model = pickle.load(training_model)      
    return(stored_model)

#------------------------- Load Translated Dataset ---------------------------#

def LoadSavedGr2En():
    greek_tweets = []
    with open('greek_files/mt_greek_dataset.txt', 'r', encoding="utf-8") as f:
        for line in f:
            greek_tweets.append(line)
    return(greek_tweets)


################################## MAIN BODY ##################################
    
if __name__ == '__main__':
    
#---------------------- Call preprocessing Functions -------------------------#
# Load for preprocessing gr2en_dataset.txt that contains the machine translated
# tweets
    
    gr2en_translated_mt_idf = LoadSavedGr2En()
    tokenized_mt_idf = Tokenize(gr2en_translated_mt_idf)[:10]
    alpha_mt_idf = NonAlphaLower(tokenized_mt_idf)
    stopwordsfree_mt_idf = removeStopwords(alpha_mt_idf)
    lemmatized_mt_idf = Lemmatizing(stopwordsfree_mt_idf)
    posTagged_mt_idf = POSTagging(lemmatized_mt_idf)

#-------------------- Call Feature Extraction Functions ----------------------#

    tweets_mt_idf, labels_mt_idf = DatasetSplit(posTagged_mt_idf)
    top_feat_mt_idf, X_mt_idf = TfIdf(tweets_mt_idf)
    tags_pct_mt_idf = POSinTopFeat(top_feat_mt_idf)

#----------------------- Call TrainTestSet Function --------------------------#

    kfold_mt_idf, X_train_mt_idf, X_test_mt_idf, y_train_mt_idf, \
    y_test_mt_idf = TrainTestSet(labels_mt_idf, X_mt_idf)
    
#------------------------ Call Classifier Functions ------------------------#

    RF_clf_mt_idf, RF_pred_mt_idf = \
    RFClassifier(X_train_mt_idf, y_train_mt_idf, X_test_mt_idf)
    MNB_clf_mt_idf, MNB_pred_mt_idf = \
    MNBClassifier(X_train_mt_idf, y_train_mt_idf, X_test_mt_idf)

#---------------------- Call Model Evaluation Function ---------------------#

    RF_confm_mt_idf, RF_cls_rep_mt_idf, RF_acc_mt_idf = \
    ModelEval(y_test_mt_idf, RF_pred_mt_idf)
    MNB_confm_mt_idf, MNB_clf_rep_mt_idf, MNB_acc_mt_idf = \
    ModelEval(y_test_mt_idf, MNB_pred_mt_idf)

#------------------------ Call Compare Algos Function ------------------------#

    all_acc, mean_acc_dev = \
    CompareAlgos(RF_clf_mt_idf, MNB_clf_mt_idf, X_mt_idf, labels_mt_idf, \
                 kfold_mt_idf)
    
#------------------------- Call Save Model Function ------------------------#

    SaveModel('saved_models/RF_classifier', RF_clf_mt_idf)
    SaveModel('saved_models/MNB_classifier', MNB_clf_mt_idf)


###############################################################################