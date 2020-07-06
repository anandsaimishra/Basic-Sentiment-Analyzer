#  Property of Godslayer™
#  Code wirtten by Anand Sai Mishra
#  On : 7/5/20, 3:52 PM

# We are using IMDB review dataset
from tensorflow.keras.datasets import imdb  #to import the imdb dataset
import numpy as np
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
# We will only use the most used 10,000 words and the rest will be ignored
print("This is the first entry on the training set :",x_train[0])
'''
Here each word is actually represented by a number value instead of the word.
As neural networks dont understand text they only understand numbers.
'''
print("These are the training labels : ", y_train[0])
'''
0 represents a negative review and 1 represents a positive review.
'''
class_names = ['Negative', 'Positive']
'''There is a helper function that can help us get the token for the corresponding word.'''
word_index = imdb.get_word_index()
'''To find the token for the word 'hello' we can use'''
print(word_index['hello'])

# Decoding the Reviews
#The decoded part is for our referance... for the decodng we use a dictionary where
# the key values paris.

reverse_word_index = dict((value, key) for key, value in word_index.items())
def decode(review):
    text = ''
    for i in review: #i means the token value of the review
        text += str(reverse_word_index[i]) # gives us the word and then
        text += ' '
    return text # this will give back the text

print(decode(x_train[0]))
# this is for our comfort to see the word instead of the tokens... though these words are not in the correct order.
'''
Before we can push this review into the neural network we have on problem ....

All these reviews are of different lengths 
'''

def show_len(apple):
    return 'Length of the first training example :', len(apple)

print(show_len(x_train[0]))