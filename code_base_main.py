import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import csv
import pandas as pd
import numpy as np
import pickle
from nltk.classify.scikitlearn import SklearnClassifier
import re
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, \
accuracy_score
from matplotlib import pyplot as plt
from collections import Counter


########################### PREPROCESSING FUNCTIONS ###########################
#--------------- Remove @users, RTs, #hashtags, urls Function ----------------#

def CleanTweets(aList):
    cleanTweetsList = []
    for review in aList:
        myStr = ' '. join(review[:-1])
        removeUser = re.sub(r"@\S+", "", myStr)
        removeRTs = re.sub(r"RT\S+", "", removeUser)
        removeURL = re.sub(r"htt\S+", "", removeRTs)
        removeHashtag = re.sub(r"#\S+", "", removeURL)
        cleanTweetsList.append([removeHashtag, review[-1]])
    return(cleanTweetsList)

#----------------------------- TOKENIZE Function -----------------------------#

def Tokenize(aList):
    tokensList = []
    for review in aList:
        tokens = word_tokenize(review)
        tokensList.append(tokens)
    return(tokensList)

#------------ Remove NON-ALPHA characters and LOWERCASE Function -------------#

def NonAlphaLower(aList):
    alphaList = []
    tempList = []
    for review in aList:
        tmpList = []
        for word in review[:-1]:
            if word.isalpha():
                tmpList.append(word.lower())
        tempList = tmpList + [review[-1]]
        alphaList.append(tempList)
    return(alphaList)

#------------- Remove STOP WORDS Function - works for lowercase! -------------#

def removeStopwords(aList):
    myStopwords = set(stopwords.words('english_adapted'))
    stopwordsFreeList = []
    for review in aList:
        tmpList = []
        for word in review[:-1]:
            if word not in myStopwords:
                tmpList.append(word)
        tempList = tmpList + [review[-1]]
        stopwordsFreeList.append(tempList)
    return(stopwordsFreeList)

#--------------------------- LEMMATIZING Function ----------------------------#

def Lemmatizing(aList):
    lm = WordNetLemmatizer()
    lemmatizedList = []
    for review in aList:
            tmpList = []
            for word in review[:-1]:
                tmpList.append(lm.lemmatize(word))
            tempList = tmpList + [review[-1]]
            lemmatizedList.append(tempList)
    return(lemmatizedList)

#------------------- POS - Part of speech tagging Function -------------------#

def POSTagging(aList):
    taggedList = []
    for review in aList:
        tagged = nltk.pos_tag(review[:-1])
        new_tagged = []
        for word in tagged:
            new_tagged.append(word[0] + '_' + word[1]) #merge word and tag
        taggedList.append(new_tagged + [review[-1]])
    return(taggedList)


######################## FEATURE EXTRACTION FUNCTIONS #########################
#-------------------------- DatasetSplit Function ----------------------------#

def DatasetSplit(aList):
    # Prepare dataset for feature extraction - Split reviews from labels
    # Split taggedList to reviewsList and labelsList
    reviewsList = []
    labelsList = []
    for review in aList:
        reviewsList.append(', '.join(review[:-1]))
        labelsList.append(review[-1])
    # Convert labels to 1s and 0s
    labelsList[:] = [1 if (label == 'Positive') else 0 for label in labelsList]
    return(reviewsList, labelsList)

#----------------------------- TF-IDF Function -------------------------------#

def TfIdf(aList):
    tf = TfidfVectorizer(max_features=1500, lowercase=False, min_df=4, \
                         max_df=0.7)
    X = tf.fit_transform(aList).toarray()
    # Sort, inverse and get indices of idf weights from more to less important
    indices = np.argsort(tf.idf_)[::-1]
    # Get features
    feature_names = tf.get_feature_names()
    # Number of features
    top_n = 100
    # Save to top_features top_n features
    top_features = [feature_names[i] for i in indices[:top_n]]
    # 5 Fold Cross-validation
    return(top_features, X)

def TrainTestSet(aList):
    Lbls = np.array(aList)
    kfold = KFold(n_splits=5, shuffle=True, random_state=1)
    for train_set,test_set in kfold.split(X):
        X_train, X_test = X[train_set], X[test_set]
        y_train, y_test = Lbls[train_set], Lbls[test_set]
    
    return(kfold, X_train, X_test, y_train, y_test)

#--------------------------- POSinTopFeat Function ---------------------------#

#% of POS tags participating in 10 top features Function
def POSinTopFeat():
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

def RFClassifier(revTrain, lblTrain, revTest):
    RF_classifier = RandomForestClassifier(n_estimators=1000, \
                                           criterion='entropy', random_state=0)
    # Train the algorithm
    RF_classifier.fit(revTrain, lblTrain)
    # Predict sentiment
    RF_predictions = RF_classifier.predict(revTest)
    return(RF_classifier, RF_predictions)
    
#---------------- Multinomial Naive Bayes Classifier Function ----------------#

def MNBClassifier(revTrain, lblTrain, revTest):
    MNB_classifier = MultinomialNB(alpha=0.5)
    # Train the algorithm
    MNB_classifier.fit(revTrain, lblTrain)
    # Predict sentiment
    MNB_predictions = MNB_classifier.predict(revTest)
    return(MNB_classifier, MNB_predictions)

#------------------------ Model Evaluation - Metrics -------------------------#

def ModelEval(lblTest, Clf_predictions):
    # Confusion matrix
    conf_matrix = confusion_matrix(lblTest, Clf_predictions)
    # Classification report
    classif_report = classification_report(lblTest, Clf_predictions)
    # Accuracy score
    accuracy = accuracy_score(lblTest, Clf_predictions)*100
    return(conf_matrix, classif_report, accuracy)

#--------------------- Compare RF - MNB algos Function -----------------------#

def CompareAlgos(classifierRF, classifierMNB):
    models = []
    results = []
    names = []
    msg = []
    evaluation = []
    models.append(('RF', classifierRF))
    models.append(('MNB', classifierMNB))
    for name, model in models:
        cv_evaluation = cross_val_score(model, X, labelsList, cv=kfold)
        results.append(cv_evaluation)
        names.append(name)
        message = (name, cv_evaluation.mean(), cv_evaluation.std())
        msg.append(message)
        evaluation.append(cv_evaluation)
    return(evaluation, msg)

#---------------------------- SaveModel Function -----------------------------#

def SaveModel(filePath, classifier):
    # Save model to filePath pickle file for direct use
    with open(filePath, 'wb') as picklefile:
        pickle.dump(classifier, picklefile)
        
#---------------------------- LoadModel Function -----------------------------#

def LoadModel(filePath):
    # Load model from filePath and store to model variable
    with open(filePath, 'rb') as training_model:
        stored_model = pickle.load(training_model)      
    # Use saved model
    predictions_sm = stored_model.predict(X_test)
    return(stored_model, predictions_sm)

#------------ Grid search to choose best hyperparameters for RF --------------#
    
def RFGridSearch(RFclassifier):
    parameters = {
            'n_estimators': [1000, 3000],
            'criterion': ['gini', 'entropy'],
            'bootstrap': [True, False]
            }
    
    grid_search = GridSearchCV(estimator = RFclassifier, param_grid = \
                               parameters, cv = 5, n_jobs = -1)
    
    grid_search.fit(X_train, y_train)
    RFbest_parameters = grid_search.best_params_
    return(RFbest_parameters)

#------------ Grid search to choose best hyperparameters for MNB -------------#

def MNBGridSearch(MNBclassifier):
    parameters = {
            'alpha': [0.1, 0.3, 0.5, 0.7, 0.9],
            'fit_prior': [True, False],
            'class_prior': [True, False]
            }
    
    grid_search = GridSearchCV(estimator = MNBclassifier, param_grid = \
                               parameters, cv = 5, n_jobs = -1)
    
    grid_search.fit(X_train, y_train)
    MNBbest_parameters = grid_search.best_params_
    return(MNBbest_parameters)


################################ MAIN BODY ####################################

if __name__ == "__main__":
    
    # LOAD reviews from myDataset_en.csv file and store to dataframe
    def LoadReviews():
        myReviewsList = []
        with open('english_files/myDataset_en.csv', 'r', \
                  encoding="ISO-8859-1", newline='') as f:
            rd  = csv.reader(f)
            for line in rd:
                myReviewsList.append(line)
        return(myReviewsList)
    
    def ReviewsToStr(aList):
    # Convert reviews to string to perform tokenize
        myStringsList = []
        for review in aList:
            myString = ' '.join(review)
            myStringsList.append(myString)
        return(myStringsList)

#---------------------- Call preprocessing Functions -------------------------#

    myReviewsList = LoadReviews()
    #print(pd.DataFrame(myReviewsList))
    cleanTweetsList = CleanTweets(myReviewsList)
    #print(pd.DataFrame(cleanTweetsList))
    myStringsList = ReviewsToStr(cleanTweetsList)
    #print(pd.DataFrame(myStringsList))
    tokensList = Tokenize(myStringsList)
    #print(pd.DataFrame(tokensList))
    alphaList = NonAlphaLower(tokensList)
    #print(pd.DataFrame(alphaList))
    stopwordsFreeList = removeStopwords(alphaList)
    #print(pd.DataFrame(stopwordsFreeList))
    lemmatizedList = Lemmatizing(stopwordsFreeList)
    #print(pd.DataFrame(lemmatizedList))
    taggedList = POSTagging(lemmatizedList)
    #print(pd.DataFrame(taggedList))

#-------------------- Call Feature Extraction Functions ----------------------#

    reviewsList, labelsList = DatasetSplit(taggedList)
    #print(pd.DataFrame(reviewsList))
    #print(pd.DataFrame(labelsList))
    top_features, X = TfIdf(reviewsList)
    #print(pd.DataFrame(top_features))
    tags_percent = POSinTopFeat()
    #print(pd.DataFrame(tags_percent))

#----------------------- Call TrainTestSet Function --------------------------#

    kfold, X_train, X_test, y_train, y_test = TrainTestSet(labelsList)

#------------------------ Call Classifier Functions ------------------------#

    RF_classifier, RF_predictions = RFClassifier(X_train, y_train, X_test)
    MNB_classifier, MNB_predictions = MNBClassifier(X_train, y_train, X_test)
    
#---------------------- Call Model Evaluation Function ---------------------#

    RFconf_matrix, RFclassif_report, RFaccuracy = \
    ModelEval(y_test, RF_predictions)
    #print(RFconf_matrix)
    #print(RFclassif_report)
    #print(RFaccuracy)
    MNBconf_matrix, MNBclassif_report, MNBaccuracy = \
    ModelEval(y_test, MNB_predictions)
    #print(MNBconf_matrix)
    #print(MNBclassif_report)
    #print(MNBaccuracy)

#------------------------ Call Compare Algos Function ------------------------#

    evaluation, msg = CompareAlgos(RF_classifier, MNB_classifier)
    
#------------------------- Call Save Model Function ------------------------#

    SaveModel('saved_models/RF_classifier', RF_classifier)
    SaveModel('saved_models/MNB_classifier', MNB_classifier)

# =============================================================================
# #--------------- Call GridSearch to find best hyperparameters ----------------#
# #------------------------- TAKES A LONG TO EXECUTE! --------------------------#
# 
#     # Call RFGridSearch
#     RFbest_parameters = RFGridSearch(RF_classifier)
#     print('RF best parameters', RFbest_parameters)
#     # Call MNBGridSearch
#     MNBbest_parameters = MNBGridSearch(MNB_classifier)
#     print('RF best parameters', MNBbest_parameters)
# =============================================================================
    MNBbest_parameters = MNBGridSearch(MNB_classifier)