import json
import csv
import pandas as pd
# LOAD, READ and save reviews files to a single file 
# Load and Label JSON file FUNCTION
def LoadJson(filepath):
    myPosList = []
    myNegList = []
    with open(filepath, 'r') as myFile:
        for line in myFile:
            myData = json.loads(line)
            if myData['overall'] == 4.0 or myData['overall'] == 5.0:
                myPosList.append([myData['reviewText'], 'Positive'])
                if len(myPosList) == 400: # number for positive reviews
                    break
        for line in myFile:
            myData = json.loads(line)
            if myData['overall'] == 1.0 or myData['overall'] == 2.0:
                myNegList.append([myData['reviewText'], 'Negative'])
                if len(myNegList) == len(myPosList): # negative equals positive
                    break
    myList = myPosList + myNegList
    return(myList)
                
# Load and Label tweets from tweets_en.csv file
with open('tweets_en.csv', encoding="ISO-8859-1") as myCSV:
    myPosTweetsList = []
    myNegTweetsList = []
    readCSV = csv.reader(myCSV, delimiter=',')
    for row in readCSV:
        if row[0] == '0':
            myNegTweetsList.append([row[5], 'Negative'])
            if len(myNegTweetsList) == 400: # number for negative reviews
                break
    for row in readCSV:
        if row[0] == '4':
            myPosTweetsList.append([row[5], 'Positive'])
            if len(myPosTweetsList) == len(myNegTweetsList): # neg equals pos
                break
myTweetsList = myPosTweetsList + myNegTweetsList
# Call LoadJson & LOAD Amazon reviews from 'Musical_Instruments.json'
# & save to myInstruList
myInstruList = LoadJson('Musical_Instruments.json')
# Call LoadJson & LOAD Amazon reviews from 'Cell_Phones_and_Accessories.json'
# & save to myCellList
myCellList = LoadJson('Cell_Phones_and_Accessories.json')
# Merge the three lists to myReviewsList
myReviewsList = myInstruList + myCellList + myTweetsList
# print(pd.DataFrame(myReviewsList))
# Save myReviewsList to myDataset_en.csv
with open('myDataset_en.csv', 'w', encoding="utf-8", newline='') as f:
    wr = csv.writer(f)
    wr.writerows(myReviewsList)