import re
import csv
from google.cloud import translate
import html


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
        removeSpaces = re.sub(r" +", " ", removeHashtag)
        removeSpace = re.sub(r"^ *", "", removeSpaces)
        cleanTweetsList.append([removeSpace, tweet[-1]])
    return(cleanTweetsList)
    
#-------------------------- Translation Function -----------------------------#
# Using Google Cloud Translation API

def gr2enTranslation(aList):
    gr2en_translated = []
    for tweet in aList:
        tmp = []
        translate = translateClient.translate(tweet[0], target_language='en')
        tmp = [translate['translatedText'], tweet[-1]]
        myStr = ' '.join(tmp)
        # decode HTML special entities in Python string
        removeHTML = html.unescape(myStr)
        gr2en_translated.append(removeHTML)
    return(gr2en_translated)
    
#---------------------- Save to mt_greek_dataset.txt -------------------------#

def Gr2En(aList):    
    with open('greek_files/mt_greek_dataset.txt', 'w', encoding="utf-8") as f:
        for tweet in aList:
            f.write('%s\n' % tweet)

if __name__ == '__main__':

    translateClient = translate.Client() # Create a client 

    
# FUNCTIONS USED TO LOAD, TRANSLATE & SAVE TWEETS TO FILE mt_greek_dataset.txt

    greek_tweets = LoadTweets()
    #print(pd.DataFrame(greek_tweets))
    cleanTweets_gr = CleanTweets(greek_tweets)
    #print(pd.DataFrame(cleanTweets_gr))
    gr2en_translated = gr2enTranslation(cleanTweets_gr)
    #print(pd.DataFrame(gr2en_translated))
    Gr2En(gr2en_translated) # save to file