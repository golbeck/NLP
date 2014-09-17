from nltk.book import *

#########################################################################################
#########################################################################################
#A concordance view shows us every occurrence of a given word, together with some context.
text1.concordance("monstrous")

#########################################################################################
#########################################################################################
#Loading your own Corpus
from nltk.corpus import PlaintextCorpusReader
#location of your file
corpus_root = '/usr/share/dict' [1]
#The second parameter of the PlaintextCorpusReader initializer can be a list of fileids, like ['a.txt', 'test/b.txt'], or a pattern that matches all fileids, like '[abc]/.*\.txt'
wordlists = PlaintextCorpusReader(corpus_root, '.*') [2]
wordlists.fileids()
wordlists.words('connectives')

#########################################################################################
#########################################################################################
# lexical resources
# see http://nltk.org/howto for more info
#stopwords, that is, high-frequency words like the, to and also that we sometimes want to filter out of a document before further processing
from nltk.corpus import stopwords
stopwords.words('english')

#filter out stopwords from a set of words labeled by 'text'
stopwords = nltk.corpus.stopwords.words('english')
content = [w for w in text if w.lower() not in stopwords]


#generate synonyms (can use to classify sentiment of a synonym)
from nltk.corpus import wordnet as wn
wn.synsets('motorcar')
wn.synset('car.n.01').lemma_names()
wn.synset('car.n.01').lemmas()


#########################################################################################
#########################################################################################
#CHAPTER 3: Processing raw text
from __future__ import division  # Python 2 users only
import nltk, re, pprint
from nltk import word_tokenize

#The variable raw contains a string with characters, including many details we are not interested in such as whitespace, line breaks and blank lines. we want to break up the string into words and punctuation. This step is called tokenization, and it produces our familiar structure, a list of words and punctuation.
tokens = word_tokenize(raw)
type(tokens)
len(tokens)
tokens[:10]
text = nltk.Text(tokens)
text.collocations()


#In order to read a local file, we need to use Python's built-in open() function, followed by the read() method
f = open('document.txt')
#The read() method creates a string with the contents of the entire file
raw = f.read()
tokens = word_tokenize(raw)
type(tokens)
words = [w.lower() for w in tokens]
type(words)
vocab = sorted(set(words))
type(vocab)

#We can also read a file one line at a time using a for loop
#'r' means to open the file for reading (the default), and 'U' stands for "Universal", which lets us ignore the different conventions used for marking newlines.
f = open('document.txt', 'rU')
for line in f:
    #we use the strip() method to remove the newline character at the end of the input line.
    print(line.strip())

#The Python codecs module provides functions to read encoded data into Unicode strings, and to write out Unicode strings in encoded form. The codecs.open() function takes an encoding parameter to specify the encoding of the file being read or written.

import codecs
f = codecs.open("file path", encoding='latin2')
#Note that we can write unicode-encoded data to a file using 
f = codecs.open(path, 'w', encoding='utf-8')

#Text read from the file object f will be returned in Unicode
for line in f:
    line = line.strip()
    print(line)
#If this does not display correctly on your terminal, or if we want to see the underlying numerical values (or "codepoints") of the characters, then we can convert all non-ASCII characters into their two-digit \xXX and four-digit \uXXXX representations:

f = codecs.open(path, encoding='latin2')
for line in f:
    line = line.strip()
    print(line.encode('unicode_escape'))

#To use regular expressions in Python we need to import the re library using: import re
import re
wordlist = [w for w in nltk.corpus.words.words('en') if w.islower()]
#Let's find words ending with ed using the regular expression «ed$». We will use the re.search(p, s) function to check whether the pattern p can be found somewhere inside the string s. We need to specify the characters of interest, and use the dollar sign which has a special behavior in the context of regular expressions in that it matches the end of the word:
[w for w in wordlist if re.search('ed$', w)]

#For some language processing tasks we want to ignore word endings, and just deal with word stems
#process word into a stem
raw = """DENNIS: Listen, strange women lying in ponds distributing swords
... is no basis for a system of government.  Supreme executive power derives from
... a mandate from the masses, not from some farcical aquatic ceremony."""
tokens = word_tokenize(raw)
#The Porter Stemmer is a good choice if you are indexing some texts and want to support search using alternative forms of words 
porter = nltk.PorterStemmer()
lancaster = nltk.LancasterStemmer()
[porter.stem(t) for t in tokens]
[lancaster.stem(t) for t in tokens]

#The WordNet lemmatizer only removes affixes if the resulting word is in its dictionary
wnl = nltk.WordNetLemmatizer()
[wnl.lemmatize(t) for t in tokens]

#Remember to prefix regular expressions with the letter r (meaning "raw"), which instructs the Python interpreter to treat the string literally, rather than processing any backslashed characters it contains.
# split this raw text on any whitespace character
re.split(r'\s+', raw)
