
from django.shortcuts import render
import math


from nltk import sent_tokenize, word_tokenize, PorterStemmer
from nltk.corpus import stopwords


from django.http import HttpResponse


def home(request):
    return render(request, "textsum.html")



def format(request):
    text_string = request.GET['t1']
    
    sizename = request.GET['t2']
    
    size = (float)(sizename)
 
    result = summarization(text_string,size)
    return render(request,"result.html",{'result':result,'text1':text_string})

def reload(request):
    return render(request,"textsum.html")




def frequency(sentences):

    frequency_dict = {}
    stopWords = set(stopwords.words("english"))
    ps = PorterStemmer()

    for sent in sentences:
        freq_table = {}
        words = word_tokenize(sent)

        for word in words:
            word = word.lower()
            word = ps.stem(word)

            if word in stopWords:
                continue 

            if word in freq_table:
                freq_table[word] += 1
   
            else:
                freq_table[word] = 1

        frequency_dict[sent] = freq_table

    return frequency_dict




def tf(freq):
    tf_dict = {}

    for sent, frequency in freq.items():
        tf_table = {}

        total_length = len(frequency)
        for word, count in frequency.items():
            tf_table[word] = count / total_length
            
        tf_dict[sent] = tf_table

    return tf_dict



def total_frequency_wordd(freq):
    frequency_word = {}

    for sent, count_dict in freq.items():

        for word, count in count_dict.items():

            if word in frequency_word:
                frequency_word[word] += 1
            else:
                frequency_word[word] = 1

    return frequency_word



def idf(freq, total_frequency_word, total_sentences):
    idf_val = {}

    for sent, idf_dict in freq.items():
        idf_table = {}

        for word in idf_dict.keys():
            
            idf_table[word] = math.log10(total_sentences / float(total_frequency_word[word]))
            
        idf_val[sent] = idf_table

    return idf_val




def tf_idf(tf_values, idf_values):
    tf_idf_val = {}

    for (sent1, freq1), (sent2, freq2) in zip(tf_values.items(), idf_values.items()):

        tf_idf_table = {}

        for (word1, value1), (word2, value2) in zip(freq1.items(), freq2.items()):

            tf_idf_table[word1] = float(value1 * value2)

        tf_idf_val[sent1] = tf_idf_table

    return tf_idf_val





def scoring(tf_idf_values) -> dict:
    
    sentence_scoring = {}

    for sent, value_dict in tf_idf_values.items():
        total_score = 0

        
        count = len(value_dict)
        for word, score in value_dict.items():
            
            total_score += score

        sentence_scoring[sent] = total_score / count

    return sentence_scoring




def thresholdd(sentence_scoring) -> int:
    
    total_score = 0
    for entry in sentence_scoring:
        total_score += sentence_scoring[entry]

    threshold = (total_score / len(sentence_scoring))

    return threshold



def summary(sentences, sentence_scoring, threshold):

    
    summary = ''

    for sentence in sentences:
        
        if sentence in sentence_scoring and sentence_scoring[sentence] >= (threshold):
            
            summary += " " + sentence
            

    return summary





def summarization(text,size):
    """
    :text: Plain text of long article (user input)
    :return: summarized text
    """

    '''
    We already have a sentence tokenizer, so we just need 
    to run the sent_tokenize() method to create the array of sentences.
    '''
    # 1 Sentence Tokenize
    sentences = sent_tokenize(text)
    total_sentences = len(sentences)
    

    # 2 Create the Frequency matrix of the words in each sentence.
    freq = frequency(sentences)



    '''
    Term frequency (TF) is how often a word appears in a sentence, divided by how many words are there in a sentence.
    '''

    # 3 Calculate TermFrequency and generate a matrix
    tf_values = tf(freq)
    


    # 4 creating table for documents per words
    total_frequency_word = total_frequency_wordd(freq)
    


    '''
    Inverse document frequency (IDF) is how unique or rare a word is.
    '''

    # 5 Calculate IDF and generate a matrix
    idf_values = idf(freq, total_frequency_word, total_sentences)
    


    # 6 Calculate TF-IDF and generate a matrix
    tf_idf_values = tf_idf(tf_values, idf_values)
    


    # 7 Score the sentences
    sentence_scoring = scoring(tf_idf_values)
    


    # 8 Find the threshold
    threshold = thresholdd(sentence_scoring)


    # 9 Generate the summary
    summaryy = summary(sentences, sentence_scoring, size * threshold)
    return summaryy



