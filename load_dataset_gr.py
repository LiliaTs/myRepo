import xlrd
import csv
import pandas as pd
import random


############################### GREEK DATASET #################################
#---------------- Load tech XLS file and Labelling Function ------------------#
def LoadTechXLS(filepath):
    myFile = xlrd.open_workbook(filepath)
    mySheet = myFile.sheet_by_index(0)
    num_rows = mySheet.nrows
    tmpList = []
    techList = []
    for row in range(0, num_rows):
        myReviews = mySheet.cell_value(row, 1)
        myRatings = mySheet.cell_value(row, 4)
        tmpList = [myReviews, myRatings]
        techList.append(tmpList)
    # Convert ratings to labels
    for review in techList:
        if review[1] == 1 or review[1] == 2:
            review[1] = 'Negative'
        elif review[1] == 4 or review[1] == 5:
            review[1] = 'Positive'         
    # Remove neutral reviews
    for review in techList:
        if review[1] == 3:
            techList.remove(review)
    return(techList)

#------------------- Load CSV file and Labelling Function --------------------#
def LoadPolCSV(filepath):
    with open(filepath, encoding="utf8") as myCSV:
        readCSV = csv.reader(myCSV, delimiter=',')
        tmpList = []
        politicalList = []
        for row in readCSV:
            #print(row)
            if row[2] != 'Neutral':
                tmpList = [row[1], row[2]]
                politicalList.append(tmpList)
    politicalList.remove(politicalList[0])
    return(politicalList)

#---------------- Load tiff XLS file and Labelling Function ------------------#

def LoadTiffXLSX(filepath):
    
    myFile = xlrd.open_workbook(filepath)
    mySheet = myFile.sheet_by_index(0)
    num_rows = mySheet.nrows
    tmpList = []
    myList = []
    for row in range(1, num_rows):
        myReviews = mySheet.cell_value(row, 4)
        myRatings = mySheet.cell_value(row, 9)
        tmpList = [myReviews, myRatings]
        myList.append(tmpList)
    # Convert ratings to labels
    negList = []
    posList = []
    for review in myList:
        if review[1] == '1':
            review[1] = 'Positive'
            posList.append(review)
        elif review[1] == '-1':
            review[1] = 'Negative'
            negList.append(review)
    random.shuffle(posList) # shuffle pos reviews &
    del posList[-489:] # remove last 489 elements to keep total neg & pos equal
    tiffList = negList + posList
    return(tiffList, negList, posList)

#------------------------------ Call Functions -------------------------------#
    
techList = LoadTechXLS('gr2en_files/sample_data_gr.xls')
politicalList = LoadPolCSV('gr2en_files/GRGE_sentiment.csv')
tiffList, negList, posList = LoadTiffXLSX('gr2en_files/tiff_gr.xlsx')
greek_reviews = techList + politicalList + tiffList
#print(pd.DataFrame(greek_reviews))

# Save myReviewsList to greek_dataset.csv
with open('gr2en_files/greek_dataset.csv', 'w', encoding="utf-8", \
          newline='') as f:
    wr = csv.writer(f)
    wr.writerows(greek_reviews)