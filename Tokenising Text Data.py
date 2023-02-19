#!/usr/bin/env python
# coding: utf-8

# # Tokenising Text Data
# In this notebook, you will learn how to tokenise text data using `tf.keras.preprocessing.text.Tokenizer`.

# In[1]:


import tensorflow as tf
tf.__version__


# You have now downloaded and experimented with the IMDb dataset of labelled movie reviews. You will have noticed that the words have been mapped to integers. Converting a sequence of words to a sequence of numbers is called _tokenisation_. The numbers themselves are called _tokens_. Tokenisation is handy because it allows numerical operations to be applied to text data.
# 
# The IMDb reviews were tokenised by mapping each word to a positive integer that indicated its frequency rank.  Tokenisation could also have been applied at the level of characters rather than words.

# ## The text dataset
# The text we will work with in this notebook is Three Men in a Boat by Jerome K. Jerome, a comical short story about the perils of going outside.

# In[2]:


# Load the data

with open('data/ThreeMenInABoat.txt', 'r', encoding='utf-8') as file:
    text_string = file.read().replace('\n', ' ')


# In[3]:


# Perform some simple preprocessing, replacing dashes with empty spaces

text_string = text_string.replace('—', '')


# In[4]:


# View an excerpt of the data

text_string[0:2001]


# In[5]:


# Split the text into sentences.

sentence_strings = text_string.split('.')


# In[6]:


# View a sample of the dataset

sentence_strings[20:30]


# ## Create a Tokenizer object

# The `Tokenizer` object allows you to easily tokenise words or characters from a text document. It has several options to allow you to adjust the tokenisation process. Documentation is available for the `Tokenizer` [here](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer).

# In[6]:


# Define any additional characters that we want to filter out (ignore) from the text

additional_filters = '—’‘“”'


# The Tokenizer has a `filters` keyword argument, that determines which characters will be filtered out from the text. The cell below shows the default characters that are filtered, to which we are adding our additional filters.

# In[10]:


# Create a Tokenizer object

from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=None, 
                      filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n' + additional_filters,
                      lower=True,
                      split=' ',
                      char_level=False,
                      oov_token='<UNK>',
                      document_count=0)


# In all, the `Tokenizer` has the following keyword arguments:
# 
# `num_words`: int. the maximum number of words to keep, based on word frequency. Only the most common `num_words-1` words will be kept. If set to `None`, all words are kept.
#     
# `filters`: str. Each element is a character that will be filtered from the texts. Defaults to all punctuation (inc. tabs and line breaks), except `'`.
# 
# `lower`: bool. Whether to convert the texts to lowercase. Defaults to `True`.
# 
# `split`: str. Separator for word splitting. Defaults to `' '`.
#     
# `char_level`: bool. if True, every character will be treated as a token. Defaults to `False`.
# 
# `oov_token`: if given, it will be added to word_index and used to replace out-of-vocabulary words during sequence_to_text calls. Defaults to `None`.

# ### Fit the Tokenizer to the text
# We can now tokenize our text using the `fit_on_texts` method. This method takes a list of strings to tokenize, as we have prepared with `sentence_strings`.

# In[11]:


# Build the Tokenizer vocabulary

tokenizer.fit_on_texts(sentence_strings)


# The `fit_on_texts` method could also take a list of lists of strings, and in this case it would recognise each element of each sublist as an individual token.

# ### Get the Tokenizer configuration
# Now that the Tokenizer has ingested the data, we can see what it has extracted from the text by viewing its configuration.

# In[12]:


# Get the tokenizer config as a python dict

tokenizer_config = tokenizer.get_config()
tokenizer_config.keys()


# In[13]:


# View the word_counts entry

tokenizer_config['word_counts']


# The above is the number of times each word appears in the corpus. As you can see, the word counts dictionaries in the config are serialized into plain JSON. The `loads()` method in the Python library `json` can be used to convert this JSON string into a dictionary.

# In[14]:


# Save the word_counts as a python dictionary

import json

word_counts = json.loads(tokenizer_config['word_counts'])


# The word index is derived from the `word_counts`. 

# In[15]:


# View the word_index entry

tokenizer_config['word_index']


# In[16]:


# Save word_index and index_word as python dictionaries

index_word = json.loads(tokenizer_config['index_word'])
word_index = json.loads(tokenizer_config['word_index'])


# ## Map the sentences to tokens
# You can map each sentence to a sequence of integer tokens using the Tokenizer's `texts_to_sequences()` method. As was the case for the IMDb data set, the number corresponding to a word is that word's frequency rank in the corpus.

# In[17]:


# View the first 5 sentences

sentence_strings[:5]


# In[20]:


# Tokenize the data

sentence_seq = tokenizer.texts_to_sequences(sentence_strings)


# In[21]:


# The return type is a list

type(sentence_seq)


# In[22]:


# View the first 5 tokenized sentences

sentence_seq[0:5]


# In[28]:


# Verify the mappings in the config

print(word_index['chapter'], word_index['i'])
print(word_index['three'], word_index['invalids'])
print(word_index['sufferings'], word_index['of'], word_index['george'], word_index['and'], word_index['harris'])
print(word_index['a'], word_index['victim'], word_index['to'], word_index['one'], word_index['hundred'], word_index['and'], word_index['seven'], word_index['fatal'], word_index['maladies'])
print(word_index['useful'], word_index['prescriptions'])


# ## Map the tokens to sentences

# You can map the tokens back to sentences using the Tokenizer's `sequences_to_texts` method.

# In[29]:


# View the first 5 tokenized sentences

sentence_seq[0:5]


# In[30]:


# Map the token sequences back to sentences

tokenizer.sequences_to_texts(sentence_seq)[:5]


# In[31]:


# Verify the mappings in the config

print(index_word['362'], index_word['8'])
print(index_word['126'], index_word['3362'])
print(index_word['2319'], index_word['6'], index_word['36'], index_word['3'], index_word['35'])
print(index_word['5'], index_word['1779'], index_word['4'], index_word['43'], index_word['363'], index_word['3'], index_word['468'], index_word['3363'], index_word['2320'])
print(index_word['2321'], index_word['3364'])


# In[32]:


# Any valid sequence of tokens can be converted to text

tokenizer.sequences_to_texts([[92, 104, 241], [152, 169, 53, 2491]])


# If a word is not featured in the Tokenizer's word index, then it will be mapped to the value of the Tokenizer's `oov_token` property. 

# In[33]:


# Tokenize unrecognised words

tokenizer.texts_to_sequences(['i would like goobleydoobly hobbledyho'])


# In[34]:


# Verify the OOV token

index_word['1']


# ## Further reading and resources
# * https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer
# * https://nlp.stanford.edu/IR-book/html/htmledition/tokenization-1.html
