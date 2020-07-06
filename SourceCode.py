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

'''
Now to solve this we can use a technique called padding.
We can use meaningless words to pad our cases.
'''
print("The token value of the word 'the' is: ",word_index['the'])
from tensorflow.keras.preprocessing.sequence import pad_sequences
x_train = pad_sequences(x_train, value = word_index['the'], padding= 'post', maxlen= 256)
x_test = pad_sequences(x_test, value = word_index['the'], padding= 'post', maxlen= 256)

print(show_len(x_train[0]))
'''
We can consider each single word in the entire dataset as a seperate feature.
The problem with that is each word will process individually and will be generalized. 
This methodology is called 'ONE HOT ENCODING' It doesnt understand the feature corelation at all.
"This Tuna Sandwich is quite tasty" -> this will not translate the learning to "This Chicken _____ is quite tasty."
'''

'''
Word Embedding: These are featre representations of various words. This is essentially that each word will have some 
values corresponding to each feature.
'''
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D

model = Sequential([
    #Embeding(layers, dimentions) layer can only be used as the first layer.
    Embedding(10000, 16),
    GlobalAveragePooling1D(), #This will convert our 10000,16 Layer to a 16 dimentional vector layer for each batch.
    Dense(16, activation='relu'),#Rectified Linear Unit. This layer has only 16 nodes
    Dense(1, activation='sigmoid') #This will give you a binary classfication output.
])

model.compile(
    loss = 'binary_crossentropy', #THis is commonly used in reclassification problems.
    optimizer = 'adam', #THis is a varient of the stocastic gradient dissent.
    metrics=['accuracy']
)
#print("Model Summery", model.summary())
model.summary()
from tensorflow.python.keras.callbacks import LambdaCallback
simple_log = LambdaCallback(on_epoch_end=lambda e, l: print(e, end='.')) #THis is to print a '.' at the end of each epoc.

E = 20 #This is the number of Epoc's that we are going to use.
h = model.fit(
    x_train, y_train,
    validation_split=0.2,
    epochs = E,
    callbacks=[simple_log],
    verbose=False
)
import matplotlib.pyplot as plt
plt.plot(range(E), h.history['accuracy'], label = 'Training')
plt.plot(range(E), h.history['val_accuracy'], label = 'Validation')
plt.legend()
plt.show()

loss, accuracy = model.evaluate(x_test, y_test)
print('Test set accuracy is : ', accuracy*100)

#Testing some predictions
p = model.predict(np.expand_dims(x_test[0], axis=0)) #If we are using the entire set then we dont need this
#But since we are only passing only one example at a time hence we need to expand the singular set.

print("The current review that we are checking is ",class_names[np.argmax(p[0])])