# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 23:40:31 2023

@author: digui
"""
# The code below demonstrates a sentiment analysis of the Avengers: Endgame movie script 
# in Python using the textblob, vader, and roberta libraries.
# The code was written by me @diguitarrista and is solely for programming demonstration purposes.
# It is not used for commercial or academic purposes
# In[ ] Libraries and list of Avengers names

import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud
from collections import Counter
from textblob import TextBlob
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import pipeline

avengers = ["TONY",
            "STEVE",
            "THOR",
            "NATASHA",
            "BRUCE",
            "CLINT",
            "SCOTT",
            "JAMES",
            "CAROL",
            "DANVERS",
            "PETER PARKER",
            "STEPHEN",
            "T'CHALLA",
            "WANDA",
            "SAM",
            "BUCKY",
            "PETER QUILL",
            "GAMORA",
            "DRAX",
            "ROCKET",
            "GROOT",
            "NEBULA",
            "MANTIS",
            "VALKYRIE",
            "KORG",
            "OKOYE",
            "WONG",
            "PEPPER",
            "HAPPY",
            "NICK",
            "HANK",
            "JANET",
            "THE ANCIENT ONE",
            "HOWARD",
            "HOPE",
            "LOKI",
            "RED",
            "THANOS",
            "EBONY",
            "PROXIMA",
            "CORVUS",
            "CULL",
            "AUNT",
            "LAURA",
            "HARLEY",
            "MIEK"
            ];


# In[ ] Read the file and separate the name of the avengers

file_path = 'avengers-endgame-script-pdf.txt'

character_lines = {}  # Use a dictionary to store character lines

with open(file_path, 'r') as file:
    lines = file.readlines()
    for i in range(len(lines)):
        line = lines[i].strip()  # Remove leading/trailing whitespace
        current_line = line

        # Check if the line is in uppercase (assumes character names are all uppercase)
        # and if it has the avenger
        for avenger in avengers:
            if line.isupper() and len(line) < 22 and avenger in current_line:
                if current_line not in character_lines:
                    character_lines[current_line] = []  # Initialize an empty list for the character

                # Append the next line to the character's lines if it's not empty
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line:
                        character_lines[current_line].append(next_line)

# In[ ] Read the file and separate the sentences for each of the avengers

avenger_lines = {}

for avenger in avengers:
    avenger_lines[avenger] = []
    for name, lines in character_lines.items():
        if avenger in name:
            for line in lines:
                avenger_lines[avenger].append(line)

# In[ ] Saves to a csv file

import csv

# Specify the CSV file name
csv_file = "avengers_lines.csv"

# Open the CSV file in write mode
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)

    # Write header row
    writer.writerow(["Avenger Name", "Line from Movie"])

    # Write data rows
    for avenger, lines in avenger_lines.items():
        for line in lines:
            writer.writerow([avenger, line])

print(f"CSV file '{csv_file}' has been created successfully.")

# In[ ] Chart of the number of sentences per avenger

key_lengths = {key: len(value) for key, value in avenger_lines.items()}

keys = list(key_lengths.keys())
lengths = list(key_lengths.values())

plt.figure(figsize=(20, 8))  

plt.bar(keys, lengths)
plt.xlabel('Avengers')
plt.ylabel('Number of lines')
plt.title('Number of lines Avengers')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

plt.tight_layout()
plt.show()

# In[ ] Word cloud chart for Iron Man

phrases = avenger_lines["TONY"]

# Combine all the phrases into a single text string
text = ' '.join(phrases)

# Tokenize the text into words
words = text.split()

# Remove stop words
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word.lower() not in stop_words]

# Count the frequency of each word
word_counts = Counter(filtered_words)

# Generate a word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_counts)

# Create a plot
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Iron Man Word Cloud Chart Avengers Endgame')

# Show the plot
plt.show()

# In[ ] Sentiment Analysis using TextBlob, VADER and RoBERTa

# TextBlob
phrase = avenger_lines["TONY"][2]
analysis_textblob = TextBlob(phrase)

#VADER

# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Perform sentiment analysis and calculate the compound score
sentiment_scores_vader = analyzer.polarity_scores(phrase)

# Get the compound sentiment score (a value between -1 and 1)
compound_score_vader = sentiment_scores_vader['compound']

# Get sentiment polarity (positive, negative, or neutral)
sentiment_textblob = analysis_textblob.sentiment.polarity

#RoBERTa
# Load the pre-trained RoBERTa model and tokenizer for sentiment analysis
model_name = "roberta-base"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name)

# Initialize a sentiment analysis pipeline
sentiment_analysis = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Perform sentiment analysis
sentiment_score_roberta = sentiment_analysis(phrase)

# In[ ] Classify sentiment scores based on the thresholds commonly used for sentiment analysis

# Define the sentiment score thresholds
threshold_textblob = 0.0  # Adjust this threshold as needed
threshold_vader = 0.05    # Adjust this threshold as needed

# Sentiment scores
textblob_polarity = sentiment_textblob
vader_scores = sentiment_scores_vader
roberta_score = sentiment_score_roberta[0]["score"]

# Classification based on TextBlob
if textblob_polarity > threshold_textblob:
    textblob_sentiment = "Positive"
elif textblob_polarity < -threshold_textblob:
    textblob_sentiment = "Negative"
else:
    textblob_sentiment = "Neutral"

# Classification based on VADER compound score
if vader_scores['compound'] >= threshold_vader:
    vader_sentiment = "Positive"
elif vader_scores['compound'] <= -threshold_vader:
    vader_sentiment = "Negative"
else:
    vader_sentiment = "Neutral"

# Classification for RoBERTa (assuming a threshold of 0.5 for positive sentiment)
if roberta_score > 0.5:
    roberta_sentiment = "Positive"
else:
    roberta_sentiment = "Negative"

# Print the entiment score
print("Tony", phrase)
print()
print("TextBlob polarity:", sentiment_textblob)
print("TextBlob Sentiment:", textblob_sentiment)
print("VADER score:", sentiment_scores_vader, compound_score_vader)
print("VADER Sentiment:", vader_sentiment)
print("RoBERTa score:", sentiment_score_roberta[0]["score"])
print("RoBERTa Sentiment:", roberta_sentiment)

# In[ ] Sentiment Analysis using TextBlob

# Initialize variables for sentiment calculation
total_polarity = 0
num_phrases = len(avenger_lines["TONY"])

# Perform sentiment analysis on each phrase and calculate total polarity
for phrase in avenger_lines["TONY"]:
    analysis = TextBlob(phrase)
    
    # Get sentiment polarity (positive, negative, or neutral)
    sentiment = analysis.sentiment.polarity
    
    # Add the polarity to the total
    total_polarity += sentiment

# Calculate the overall sentiment score
overall_sentiment = total_polarity / num_phrases

# Define sentiment labels based on the overall score
if overall_sentiment > 0:
    overall_sentiment_label = "Positive"
elif overall_sentiment < 0:
    overall_sentiment_label = "Negative"
else:
    overall_sentiment_label = "Neutral"

# Print the overall sentiment score
print(f"Overall Sentiment: {overall_sentiment_label} (Score: {overall_sentiment})")


# In[ ] Sentiment Analysis using VADER

# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Initialize variables for sentiment calculation
total_compound_score = 0

# Perform sentiment analysis on each phrase and calculate total compound score
for phrase in avenger_lines["TONY"]:
    sentiment_scores = analyzer.polarity_scores(phrase)
    
    # Get the compound sentiment score (a value between -1 and 1)
    compound_score = sentiment_scores['compound']
    
    # Add the compound score to the total
    total_compound_score += compound_score

# Calculate the overall sentiment score
overall_sentiment = total_compound_score / len(avenger_lines["TONY"])

# Define sentiment labels based on the overall score
if overall_sentiment > 0:
    overall_sentiment_label = "Positive"
elif overall_sentiment < 0:
    overall_sentiment_label = "Negative"
else:
    overall_sentiment_label = "Neutral"

# Print the overall sentiment score
print(f"Overall Sentiment: {overall_sentiment_label} (Score: {overall_sentiment})")


# In[ ] Sentiment Analysis using Roberta

# Load the pre-trained RoBERTa model and tokenizer for sentiment analysis
model_name = "roberta-base"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name)

# Initialize a sentiment analysis pipeline
sentiment_analysis = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Initialize variables to accumulate sentiment scores
total_sentiment_score = 0.0
num_phrases = len(avenger_lines["TONY"])

# Perform sentiment analysis on each phrase and accumulate the scores
for phrase in avenger_lines["TONY"]:
    results = sentiment_analysis(phrase)
    sentiment_score = results[0]["score"]
    total_sentiment_score += sentiment_score

# Calculate the average sentiment score
average_sentiment_score = total_sentiment_score / num_phrases

# Determine the overall sentiment based on the average score
overall_sentiment = "POSITIVE" if average_sentiment_score > 0.5 else "NEGATIVE"

# Print the overall sentiment result
print(f"Overall Sentiment for TONY's phrases: {overall_sentiment}")
print(f"Average Sentiment Score: {average_sentiment_score}")

# In[ ] Sentiment Analysis using TextBlob of all lines for each avenger

# Initialize the avengers_sentimental_textblob dictionary
avengers_sentimental_textblob = {}
avengers_sentimental_textblob_avarage_score = 0 

# Iterate through each character's lines in the avenger_lines dictionary
for character, lines in avenger_lines.items():
    # Initialize variables for sentiment calculation for each character
    total_polarity = 0
    num_phrases = len(lines)

    # Perform sentiment analysis on each phrase and calculate total polarity
    for phrase in lines:
        analysis = TextBlob(phrase)

        # Get sentiment polarity (positive, negative, or neutral)
        sentiment = analysis.sentiment.polarity

        # Add the polarity to the total
        total_polarity += sentiment

    # Calculate the overall sentiment score for the character's lines
    if total_polarity > 0:
        overall_sentiment = total_polarity / num_phrases
    else: 
        overall_sentiment = 0
        
    # Determine the overall sentiment based on the average score
    if overall_sentiment > 0:
        overall_sentiment_label = "POSITIVE" 
    elif overall_sentiment == 0:
        overall_sentiment_label = "NEUTRAL" 
    else: 
        overall_sentiment_label = "NEGATIVE"

    # Store the character's sentiment in the avengers_sentimental_textblob dictionary
    avengers_sentimental_textblob[character] = {
        "Overall Sentiment": overall_sentiment_label,
        "Sentiment Score": overall_sentiment
    }
    
    avengers_sentimental_textblob_avarage_score += overall_sentiment / len(avenger_lines.keys())
# In[ ] Convert the dicitonary avengers_sentimental_textblob into a dataframe

# Create a DataFrame from the avengers_sentimental_textblob dictionary
df = pd.DataFrame(avengers_sentimental_textblob).T

# In[ ] Chart of the average sentiment score for each avenger

# Adjust the figure size
plt.figure(figsize=(14, 8))  # Change the width (12) and height (6) as needed

# Chart for Sentiment Score versus Avenger
df['Sentiment Score'].plot(kind='bar', color='lightcoral')
plt.title('Sentiment Score by Avenger (TextBlob)')
plt.xlabel('Avenger')
plt.ylabel('Sentiment Score')
plt.xticks(rotation=45)
plt.show()

# In[ ] Chart of the overall sentiments

# Frequency chart for Overall Sentiment
sentiment_counts = df['Overall Sentiment'].value_counts()
sentiment_counts.plot(kind='bar', color='skyblue')
plt.title('Overall Sentiment Frequency (TextBlob)')
plt.xlabel('Sentiment')
plt.ylabel('Frequency')
plt.show()

# In[ ] Sentiment Analysis using VADER of all lines for each avenger

# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Initialize a dictionary to store sentiment scores for each character
avengers_sentimental_vader = {}
avengers_sentimental_vader_avarage_score = 0

# Loop through all keys in the avenger_lines dictionary
for character, lines in avenger_lines.items():
    total_compound_score = 0

    # Perform sentiment analysis on each phrase for the current character
    for phrase in lines:
        sentiment_scores = analyzer.polarity_scores(phrase)

        # Get the compound sentiment score (a value between -1 and 1)
        compound_score = sentiment_scores['compound']

        # Add the compound score to the total
        total_compound_score += compound_score

    # Calculate the overall sentiment score for the current character
    if total_compound_score > 0:
        overall_sentiment = total_compound_score / len(lines)
    else:
        overall_sentiment = 0

    # Determine the overall sentiment based on the average score
    if overall_sentiment > 0:
        overall_sentiment_label = "POSITIVE" 
    elif overall_sentiment == 0:
        overall_sentiment_label = "NEUTRAL" 
    else: 
        overall_sentiment_label = "NEGATIVE"

    # Store the overall sentiment score and label in the new dictionary
    avengers_sentimental_vader[character] = {
        "Overall Sentiment": overall_sentiment_label,
        "Sentiment Score": overall_sentiment
    }
    
    avengers_sentimental_vader_avarage_score += overall_sentiment / len(avenger_lines.keys())
    
# In[ ] Chart of the average sentiment score for each avenger

# Create lists to store sentiment labels and scores
sentiment_labels = []
sentiment_scores = []

# Extract sentiment information from the avengers_sentimental_vader dictionary
for character, sentiment in avengers_sentimental_vader.items():
    sentiment_labels.append(sentiment['Overall Sentiment'])
    sentiment_scores.append(sentiment['Sentiment Score'])

# Create a bar chart for sentiment scores by Avenger
plt.figure(figsize=(12, 6))
plt.bar(avengers_sentimental_vader.keys(), sentiment_scores, color='salmon')
plt.title('Sentiment Scores by Avenger (Vader)')
plt.xlabel('Avenger')
plt.ylabel('Sentiment Score')
plt.xticks(rotation=45)
plt.show()

# In[ ] Chart of the overall sentiments

# Create a frequency chart for Overall Sentiment labels
plt.figure(figsize=(8, 6))
plt.hist(sentiment_labels, bins=['POSITIVE', 'NEUTRAL', 'NEGATIVE'], alpha=0.7, color='skyblue')
plt.title('Overall Sentiment Frequency (Vader)')
plt.xlabel('Sentiment Labels')
plt.ylabel('Frequency')
plt.show()

# In[ ] Sentiment Analysis using Roberta of all lines for each avenger

from collections import defaultdict

# Create a dictionary to store cumulative sentiment scores
avengers_sentimental_roberta = defaultdict(float)
avengers_sentimental_roberta_avarage_score = 0

# Iterate through each key in avenger_lines
for key, phrases in avenger_lines.items():
    total_sentiment_score = 0.0
    num_phrases = len(phrases)

    # Perform sentiment analysis on each phrase and accumulate the scores
    for phrase in phrases:
        results = sentiment_analysis(phrase)
        sentiment_score = results[0]["score"]
        total_sentiment_score += sentiment_score
    
    # Calculate the average sentiment score
    if total_sentiment_score > 0: 
        average_sentiment_score = total_sentiment_score / num_phrases
    else:
        average_sentiment_score = 0

    # Determine the overall sentiment based on the average score
    if average_sentiment_score > 0.5:
        overall_sentiment = "POSITIVE" 
    elif average_sentiment_score >= 0.2:
        overall_sentiment = "NEUTRAL" 
    else: 
        overall_sentiment = "NEGATIVE"
        
    # Store the cumulative sentiment score for the current key
    avengers_sentimental_roberta[key] = [average_sentiment_score, overall_sentiment]
    
    avengers_sentimental_roberta_avarage_score += average_sentiment_score / len(avenger_lines.keys())
# In[ ] Chart of the average sentiment score for each avenger

# Extract Avenger names, average sentiment scores, and overall sentiment scores
avengers = list(avengers_sentimental_roberta.keys())
average_sentiments = [entry[0] for entry in avengers_sentimental_roberta.values()]

# Create the first chart for average_sentiment_score
plt.figure(figsize=(14, 8))
plt.bar(avengers, average_sentiments, color='skyblue')
plt.xlabel('Avenger')
plt.ylabel('Average Sentiment Score')
plt.title('Average Sentiment Score per Avenger (Roberta)')
plt.xticks(rotation=45)
plt.tight_layout()

# Show the first chart
plt.show()

# In[ ] Chart of the overall sentiments

# Extract overall sentiments
overall_sentiments = [entry[1] for entry in avengers_sentimental_roberta.values()]

# Count the occurrences of each sentiment
sentiment_counts = Counter(overall_sentiments)

# Define the sentiment labels and counts
sentiments = list(sentiment_counts.keys())
counts = list(sentiment_counts.values())

# Create a bar chart
plt.figure(figsize=(8, 5))
plt.bar(sentiments, counts, color=['lightgreen', 'lightcoral', 'lightskyblue'])
plt.xlabel('Sentiment')
plt.ylabel('Frequency')
plt.title('Sentiment Distribution Among Avengers (Roberta)')
plt.tight_layout()

# Show the chart
plt.show()

# In[ ] Sentimental analysis including values ​​equal to zero

# Data
data = {
    'Method': ['Roberta', 'Vader', 'TextBlob'],
    'Average Score': [avengers_sentimental_roberta_avarage_score, 
                      avengers_sentimental_vader_avarage_score, 
                      avengers_sentimental_textblob_avarage_score]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Function to determine sentiment
def determine_sentiment(score):
    if data['Method'] == 'TextBlob' or data['Method'] == 'Vader':
        if data['Average Score'] > 0: 
            return 'Positive'
        elif data['Average Score'] == 0:
            return 'Neutral'
        else:
            return 'Negative'
    else:
        if data['Average Score'][0] > 0.5: 
            return 'Positive'
        elif data['Average Score'][0] >= 0.2:
            return 'Neutral'
        else:
            return 'Negative'

# Apply the sentiment determination function to the 'Average Score' column
df['Sentiment'] = df['Average Score'].apply(determine_sentiment)

# Plot the table
plt.figure(figsize=(8, 4))
plt.axis('off')  # Hide axis
plt.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center', colColours=['#f2f2f2']*df.shape[1])
plt.title('Sentiment Analysis of Avengers Reviews')
plt.show()