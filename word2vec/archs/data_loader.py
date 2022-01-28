import sys
sys.path.insert(0, "../")

import tensorflow as tf
import tensorflow_datasets as tfds

import nltk
from nltk.corpus import stopwords
    
from typing import List, Any, Dict

import tqdm

from word2vec.archs.constants import (CBOW_N_WORDS, SKIPGRAM_N_WORDS, MIN_WORD_FREQUENCY, MAX_SEQ_LENGTH, 
                                      BATCH_SIZE, VOCAB_SIZE, BUFFER_SIZE, N_ELEMNT, LANGUAGE)
from word2vec.tools.timer import timeit, SpeedTest

try:
    STOPWORDS = set(stopwords.words(LANGUAGE))
    STOPWORDS_TENSOR = list(STOPWORDS)
except LookupError:
    import nltk
    nltk.download('stopwords')
    STOPWORDS = set(stopwords.words(LANGUAGE))





# ******************** global constants ******************

AUTOTUNE = tf.data.AUTOTUNE





# ******************** tools *************************

def tf_isin(e: tf.Tensor, big_tensor: tf.Tensor) -> tf.bool:

    return tf.math.reduce_any(tf.math.equal(e, big_tensor))




@tf.function
def remove_stop_words(rtensor: tf.Tensor) -> tf.Tensor:

    ta = tf.TensorArray(tf.bool, size = 0, dynamic_size = True, clear_after_read = True)
    i = 0
    
    for w in rtensor:

        ta = ta.write(i, tf.math.logical_not(tf_isin(w, STOPWORDS_TENSOR)))
        i += 1
    
    msk = ta.stack()
    
    return rtensor[msk]





# ********************* loading *************************

class DataLoader:

    def __init__(self, name: str) -> None:
        """[summary]
        the datasets are always in bytes, but sometimes the lists returned are in string like combine_first_strings
        
        Args:
            path ([type]): [description]
            language ([type]): ex : 'english'
        """
        self.name = name
        self.stopwords_tensor = tf.constant(list(STOPWORDS))
        
        
    def load(self) -> None:
        """
        self.ds : the dataset, can be lowered, but never split
            tf.Tensor: shape=(x,), dtype=string, where x varies
            
        self.ds_split : the dataset split
            tf.Tensor: shape=(), dtype=string
            
        """
    
        self.ds = tfds.load(name = self.name, split = 'train').take(N_ELEMNT)
        print(f'cardinality : {self.ds.cardinality()}')
        
        self.ds_split = None
        
        
    @timeit  
    def prepare(self, lower: bool = True, split: bool = True) -> tf.data.Dataset:
        """
        The function 'remove...' must be adapted to the dataset, given the format of the data
        
        Here for instance it does the following:
        remove the \n and select x['text']
        splits the strings
        maps the replacements ie :
            remove the @-@ symbol
            remove first and last spaces
            remove multiple spaces
        filters empty strings
        filters string of one caracter

        caches and prefetchs for speed

        
        (remove_symbols is equivalent of 'isalpha()')
        
        Args:
            lower (bool, optional): [description]. Defaults to True.
            split (bool, optional): [description]. Defaults to True.
        """
        if not hasattr(self, 'ds'):
            self.load()
    
        def content_filter(x: tf.Tensor) -> tf.Tensor:
            """[summary]
            Delete all the string of align, dash, and presentation to only keep text
            Args:
                x (tf.Tensor): [description]

            Returns:
                tf.Tensor: [description]
            """

            return tf.logical_not(tf.strings.regex_full_match(x, '([[:space:]][=])+.+([[:space:]][=])+[[:space:]]*'))

        def remove_the_at(x: tf.Tensor) -> tf.Tensor:

            return tf.strings.regex_replace(x, '@-@', '', replace_global=True)

        def remove_punctuation(x: tf.Tensor) -> tf.Tensor:
            """[summary]
            The '\' symbol is let in the text because it helps to code french accents in utf-8
            Args:
                x (tf.Tensor): [description]

            Returns:
                tf.Tensor: [description]
            """
        
            return tf.strings.regex_replace(x, "[.,/#!$%\^&\*;:{}=\-_`~()]", '', replace_global=True)

        def remove_multiple_spaces(x: tf.Tensor) -> tf.Tensor:

            return tf.strings.regex_replace(x, ' +', ' ', replace_global=True)
        
        def remove_first_and_last_spaces(x: tf.Tensor) -> tf.Tensor:
            
            return tf.strings.regex_replace(x, "^\s*|\s*$", '', replace_global=True)

        def all_mapping_in_one(x: tf.Tensor) -> tf.Tensor:
            
            return remove_first_and_last_spaces(remove_multiple_spaces(remove_punctuation(remove_the_at(x))))
        
        def remove_align(x: Dict[str, tf.Tensor]) -> tf.Tensor:

            return tf.strings.regex_replace(x['text'], '\n', ' ', replace_global=True)

        # ************* basic treatment ************
        
        
        ds = self.ds.map(remove_align, num_parallel_calls = AUTOTUNE) # select x['text'] and replace \n by ' '
        
        ds = ds.map(lambda x: tf.strings.split(x, '. '), num_parallel_calls = AUTOTUNE) # split paragraph into sentences, and unbatch to finally get a 
        ds = ds.unbatch().cache().prefetch(AUTOTUNE)                                    # ds of sentences
        
        ds = ds.map(all_mapping_in_one, num_parallel_calls = AUTOTUNE)

        ds = ds.filter(lambda x: tf.cast(tf.strings.length(x), bool))
        ds = ds.filter(lambda x: tf.cast(tf.strings.length(x)-1, bool))
        self.ds = ds.filter(content_filter)
        
        # ************* additional treatment ************

        if lower:
            self.lower()
                
        if split:
            self.split()
                
        return self.ds_split if self.ds_split is not None else self.ds
    
    
    def prepare_old(self, lower: bool = True, split: bool = True) -> tf.data.Dataset:
        """
        old_version
        
        Args:
            lower (bool, optional): [description]. Defaults to True.
            split (bool, optional): [description]. Defaults to True.
        """
        
        if not hasattr(self, 'ds'):
            self.load()
    
        def content_filter(x: tf.Tensor) -> tf.Tensor:

            return tf.logical_not(tf.strings.regex_full_match(x, '([[:space:]][=])+.+([[:space:]][=])+[[:space:]]*'))

        def remove_the_at(x: tf.Tensor) -> tf.Tensor:

            return tf.strings.regex_replace(x, '@-@', '', replace_global=True)

        def remove_symbols(x: tf.Tensor) -> tf.Tensor:

            return tf.strings.regex_replace(x, '[^a-zA-Z ]', '', replace_global=True)

        def remove_multiple_spaces(x: tf.Tensor) -> tf.Tensor:

            return tf.strings.regex_replace(x, ' +', ' ', replace_global=True)
        
        def remove_first_and_last_spaces(x: tf.Tensor) -> tf.Tensor:
            
            return tf.strings.regex_replace(x, "^\s*|\s*$", '', replace_global=True)

        def all_mapping_in_one(x: tf.Tensor) -> tf.Tensor:
            
            return remove_first_and_last_spaces(remove_multiple_spaces(remove_symbols(remove_the_at(x))))

        # ************* basic treatment ************
        
        ds = self.ds.map(lambda x: tf.strings.split(x, ' . '))
        ds = ds.map(all_mapping_in_one, num_parallel_calls = AUTOTUNE)

        ds = ds.unbatch().cache().prefetch(AUTOTUNE)

        ds = ds.filter(lambda x: tf.cast(tf.strings.length(x), bool))
        ds = ds.filter(lambda x: tf.cast(tf.strings.length(x)-1, bool))
        self.ds = ds.filter(content_filter)
        
        # ************* additional treatment ************
        
        if lower:
            self.lower()
                
        if split:
            self.split()
                
        return self.ds_split if self.ds_split is not None else self.ds
    
    
    def lower(self) -> None:
        
        self.ds = self.ds.map(lambda s : tf.strings.lower(s, encoding = 'utf-8'), num_parallel_calls = AUTOTUNE)
        
        
    def split(self) -> None:
        
        self.ds_split = self.ds.map(lambda s : tf.strings.split(s, sep = ' '), num_parallel_calls = AUTOTUNE)
    
    
    @timeit
    def without_stopwords(self) -> tf.data.Dataset:
        
        self.ds = self.ds.map(lambda x : tf.strings.split(x, sep = ' '))
        self.ds = self.ds.map(remove_stop_words)
        self.ds = self.ds.map(lambda t : tf.strings.reduce_join(t, separator = ' ')).cache().prefetch(AUTOTUNE)
        
        return self.ds




def pipeline_cbow(tokenized_sequence: tf.Tensor) -> tf.Tensor:
    #TODO : mieux tester cette fonction
    """
    return the tensor of each [context, target] pair
    where context is made of N = CBOW_N_WORDS past words and N = CBOW_N_WORDS future words
    target is the word in the middle.
    
    Long paragraphs will be troncated into paragraphs of length MAX_SEQ_LENGTH. 

    Each element in `context` is N=CBOW_N_WORDS*2 context words.
    Each element in `target` is a middle word.
    
    Args:
        tokenized_sequence ([type]): tokenized_sequence is a tensor of one tokenized sequences.
    """
    
    ta = tf.TensorArray(tf.int64, size = 0, dynamic_size = True, clear_after_read = False)
    i = 0
    
    
    if MAX_SEQ_LENGTH:
        tokenized_sequence = tokenized_sequence[:MAX_SEQ_LENGTH]
        
    msk = tf.range(start = 0, limit = 2 * SKIPGRAM_N_WORDS + 1)
    msk = tf.math.equal(msk, SKIPGRAM_N_WORDS)
    msk = tf.math.logical_not(msk)
    
    for idx in range(tf.shape(tokenized_sequence)[0])[CBOW_N_WORDS: -CBOW_N_WORDS]:
        window = tokenized_sequence[idx - CBOW_N_WORDS: idx + CBOW_N_WORDS + 1]
        y = window[SKIPGRAM_N_WORDS]
        x = window[msk]
        
        ta = ta.write(i, (x, y))
        i += 1
    
    return ta.stack()




def pipeline_skipgram(tokenized_sequence: tf.Tensor) -> tf.Tensor:
    """
    return the tensor of each [context, target] pair
    where context is made of N = CBOW_N_WORDS past words and N = CBOW_N_WORDS future words
    target is the word in the middle.
    
    Long paragraphs will be troncated into paragraphs of length MAX_SEQ_LENGTH. 

    Each element in `context` is ONE context word (each target word is paired several times to each context word in the third 'for').
    Each element in `target` is a middle word.
    
    /!\ the 'target' will be fed as the input of the skip-gram word2vec, and the model will try try to predict the 'context' 
    
    Args:
        tokenized_sequence ([type]): tokenized_sequence is a tensor of one tokenized sequences.
    """
    
    ta = tf.TensorArray(tf.int64, size = 0, dynamic_size = True, clear_after_read = False)
    i = 0
    
    
    if MAX_SEQ_LENGTH:
        tokenized_sequence = tokenized_sequence[:MAX_SEQ_LENGTH]
        
    msk = tf.range(start = 0, limit = 2 * SKIPGRAM_N_WORDS + 1)
    msk = tf.math.equal(msk, SKIPGRAM_N_WORDS)
    msk = tf.math.logical_not(msk)
    
    for idx in range(tf.shape(tokenized_sequence)[0])[SKIPGRAM_N_WORDS: -SKIPGRAM_N_WORDS]:
        window = tokenized_sequence[idx - SKIPGRAM_N_WORDS: idx + SKIPGRAM_N_WORDS + 1]
        x = window[SKIPGRAM_N_WORDS]
        y_ = window[msk]
        
        for y in y_:
            ta = ta.write(i, (x, y))
            i += 1
    
    return ta.stack()




class Word2Int:
    
    def __init__(self) -> None:
        
        self.vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(standardize = None,
                                                                              max_tokens = VOCAB_SIZE,
                                                                              output_mode = 'int'
                                                                              )
    
    
    @timeit
    def adapt(self, ds):
        """Takes a batched ds

        Args:
            ds ([type]): [description]
        """
        
        self.vectorize_layer.adapt(ds.batch(batch_size = BATCH_SIZE))
        self.inverse_vocab = self.vectorize_layer.get_vocabulary() # l'indice est l'entier associé, le mot le plus fréquent etant en premier
    
    
    @timeit    
    def vectorize(self, ds):
        
        if not hasattr(self, 'inverse_vocab'):
            self.adapt(ds)
        
        self.text_vector_ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE).map(self.vectorize_layer).unbatch()
        
        return self.text_vector_ds
    
    
    def get_vocab(self):
        
        if not hasattr(self, 'inverse_vocab'):
            pass
        
        else:
            return self.inverse_vocab
        
        


class GetDataset:
    
    def __init__(self, type_model, path: str) -> None:
        
        assert(type_model in ['skip_gram', 'cbow'])
        self.type_model = type_model
        
        self.w2i = Word2Int()
        self.dl = DataLoader(path)
        self.pipeline_fn = pipeline_skipgram if self.type_model == 'skip_gram' else pipeline_cbow
    
    
    def get_ds_ready(self) -> tf.data.Dataset:
        
        print(f"tf.__version__: {tf.__version__}")
        print()
        
        print('.... Start preparing dataset')
        self.dl.prepare()
        print('done;')
        print()
        
        print('.... Start removing stopwords')
        text_ds = self.dl.without_stopwords()
        print(f'kind of dataset : {text_ds}')
        print('done;')
        print()
        
        print('.... Adapting to TextVectorization')
        self.w2i.adapt(text_ds)
        self.vocab = self.w2i.inverse_vocab
        print('done;')
        print()
        
        print('.... Vectorization')
        text_vector_ds = self.w2i.vectorize(text_ds)
        print('done;')
        print()
        
        print('.... Filtering the dataset')
        with SpeedTest('filter'):     
            ds = text_vector_ds.filter(lambda x : tf.math.greater_equal(tf.shape(x)[0], 2 * SKIPGRAM_N_WORDS + 1))
        print('done;')
        print()
        
        print('.... Mapping the pipeline function')
        with SpeedTest('map'):
            ds = text_vector_ds.map(self.pipeline_fn)
        print('done;')
        print()
        
        print('.... Unbatching')
        with SpeedTest('unbatch'):    
            ds = ds.unbatch()
        print('done;')
        print()
            
        print('.... Filtering the [0, 0] tensors')
        with SpeedTest('filter'):     
            ds = ds.filter(lambda x : tf.math.logical_not(tf.math.reduce_all(tf.math.equal(x, tf.constant([0, 0], dtype = tf.int64)))))
        print('done;')
        print()
        
        print('.... Turning into tuples')
        with SpeedTest('tuples'):
            ds = ds.map(lambda t: (t[0], t[1]))
        print('done;')
        print()
        
        return ds
        
    
    def get_vocab(self):
        
        if not hasattr(self.w2i, 'inverse_vocab'):
            raise ValueError('w2i not adapted to a dataset')
        
        return self.w2i.inverse_vocab