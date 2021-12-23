import sys
sys.path.insert(0, "../")

import tensorflow as tf

try:
    from word2vec.archs.constants import CBOW_N_WORDS, SKIPGRAM_N_WORDS, MIN_WORD_FREQUENCY, MAX_SEQ_LENGTH, BATCH_SIZE, VOCAB_SIZE, BUFFER_SIZE
except ImportError:
    raise ImportError('constants' + ' non importé')

import nltk
from nltk.corpus import stopwords
    
from typing import List, Any

import tqdm

try:
    import word2vec.archs.constants as c
except ImportError:
    c = None
    raise ImportError('constants' + ' not imported')

try:
    STOPWORDS = set(stopwords.words(c.LANGUAGE))
except LookupError:
    import nltk
    nltk.download('stopwords')
    STOPWORDS = set(stopwords.words(c.LANGUAGE))


# ******************** global constants ******************

AUTOTUNE = tf.data.AUTOTUNE





# ********************* loading *************************

class DataLoader:

    def __init__(self, path: str) -> None:
        """[summary]
        the datasets are always in bytes, but sometimes the lists returned are in string like combine_first_strings
        
        Args:
            path ([type]): [description]
            language ([type]): ex : 'english'
        """
        self.path = path
        
        
    def load(self) -> None:
        """
        self.ds : the dataset, can be lowered, but never split
            tf.Tensor: shape=(x,), dtype=string, where x varies
            
        self.ds_split : the dataset split
            tf.Tensor: shape=(), dtype=string
            
        """
    
        self.ds = tf.data.TextLineDataset(self.path)
        
        self.ds_split = None
        
    
    def prepare(self, lower: bool = True, split: bool = True) -> tf.data.Dataset:
        """
        TODO faire en sorte qu'il y ait un seul mapping au lieu de 3 -> plus vite
        TODO traduire en anglais
        TODO Fonction à adapter en fonction de la dataset, marche ici pour une ds wikipédia classique
        split les strings
        fait le mapping des remplacements ie :
            enleve le symbole @-@ qui est un peu partout, surement les hyperliens je pense, mais qui ous est inutile
            eleve ce qui n'est pas une lettre (symboles chinois etc)
            enleve les espaces multiples et les remplace par un seul espace
        filtre les strings vides
        filtre les strings d'un caractère (transition entre paragraphe)
        filtre les titres de paragraphes wikipedia

        cache et prefetch pour plus de rapidité

        
        remove_symbols is equivalent of 'isalpha()'
        
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
    
    def without_stopwords(self, n_strings: int) -> List[str]:
        """when iterating over tf.Dataset, you can't iterate over one element with a comprehension list or tf.map_fn
        to remove the stopwords, we have to do it from the combined_strings

        Args:
            n_strings ([type]): [description]
        Returns:
            string
        """     
        combined_strings = self.combine_first_strings(n_strings)
        word_list = combined_strings.split()
        word_list_without_stop_words = [w for w in word_list if w not in STOPWORDS]
        
        return word_list_without_stop_words    

    def combine_first_strings(self, n_strings: int) -> str:
        """useful for wordcloud

        Args:
            n_strings ([type]): [description]
        Returns:
            string
        """
        combined_strings = ' '.join([s.numpy().decode('utf-8') for s in self.ds.take(n_strings).__iter__()])
        
        return combined_strings
    
    def combine_first_strings_all_ds(self) -> bytes:
        """[summary]

        Args:
            n_strings ([type]): [description]
        """
        combined_strings = self.ds.reduce('', lambda s, s_: s + ' ' + s_)
        
        return combined_strings
    
    def list_of_sentences_without_stopwords(self) -> List[List[str]]:
        """[summary]

        Args:
            n_strings ([type]): [description]
        """
        combined_strings_coma_separated = ', '.join([s.numpy().decode('utf-8') for s in self.ds.__iter__()])
        list_of_sentences = combined_strings_coma_separated.split(sep = ', ')
        list_of_sentences_split = [[w for w in s.split() if w not in STOPWORDS] for s in list_of_sentences]
        
        return list_of_sentences_split
    
    def list_of_sentences_without_stopwords_not_split(self) -> List[List[str]]:
        """[summary]

        Args:
            n_strings ([type]): [description]
        """                
        return [' '.join([w for w in s]) for s in self.list_of_sentences_without_stopwords()]



def pipeline_cbow(batch):
    """
    return (context, target)
    where context is made of N = CBOW_N_WORDS past words and N = CBOW_N_WORDS future words
    target is the word in the middle.
    
    Long paragraphs will be troncated into paragraphs of length MAX_SEQ_LENGTH. 

    Each element in `context` is N=CBOW_N_WORDS*2 context words.
    Each element in `target` is a middle word.
    
    Args:
        b ([type]): b is a batch (list) of tokenized sequences.
    """
    
    context, target = [], []
    
    for tokenized_sequence in tqdm.tqdm(batch):
        
        if len(tokenized_sequence) < SKIPGRAM_N_WORDS * 2 + 1: # we do not take this sentence into account
            continue
        
        if MAX_SEQ_LENGTH:
            tokenized_sequence = tokenized_sequence[:MAX_SEQ_LENGTH]
        
        for idx in range(len(tokenized_sequence))[CBOW_N_WORDS: -CBOW_N_WORDS]:
            window = tokenized_sequence[idx - CBOW_N_WORDS: idx + CBOW_N_WORDS + 1]
            y = window.pop(CBOW_N_WORDS)
            x = window # once the pop is done
            context.append(x)
            target.append(y)
    
    return context, target




def pipeline_skipgram(batch):
    """
    return (context, target)
    where context is made of N = CBOW_N_WORDS past words and N = CBOW_N_WORDS future words
    target is the word in the middle.
    
    Long paragraphs will be troncated into paragraphs of length MAX_SEQ_LENGTH. 

    Each element in `context` is ONE context word (each target word is paired several times to each context word in the third 'for').
    Each element in `target` is a middle word.
    
    /!\ the 'target' will be fed as the input of the skip-gram word2vec, and the model will try try to predict the 'context' 
    
    Args:
        b ([type]): b is a batch (list) of tokenized sequences.
    """
    
    context, target = [], []
    
    for tokenized_sequence in tqdm.tqdm(batch):
        
        if len(tokenized_sequence) < SKIPGRAM_N_WORDS * 2 + 1: # we do not take this sentence into account
            continue
        
        if MAX_SEQ_LENGTH:
            tokenized_sequence = tokenized_sequence[:MAX_SEQ_LENGTH]
        
        for idx in range(len(tokenized_sequence))[CBOW_N_WORDS: -CBOW_N_WORDS]:
            window = tokenized_sequence[idx - CBOW_N_WORDS: idx + CBOW_N_WORDS + 1]
            x = window.pop(CBOW_N_WORDS)
            y_ = window # once the pop is done
            
            for y in y_:
                context.append(y)
                target.append(x)
    
    return context, target




class Word2Int:
    
    def __init__(self) -> None:
        
        self.vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(standardize = None,
                                                                              max_tokens = VOCAB_SIZE,
                                                                              output_mode = 'int'
                                                                              )
    
    def adapt(self, ds):
        """Takes a batched ds

        Args:
            ds ([type]): [description]
        """
        
        self.vectorize_layer.adapt(ds)
        self.inverse_vocab = self.vectorize_layer.get_vocabulary() # l'indice est l'entier associé, le mot le plus fréquent etant en premier
        
    def vectorize(self, ds):
        
        if not hasattr(self, 'inverse_vocab'):
            self.adapt(ds)
        
        self.text_vector_ds = ds.map(self.vectorize_layer)
        
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

    
    def get_ds_ready(self):
        
        return self.get_ds_ready_skip_gram() if self.type_model == 'skip_gram' else self.get_ds_ready_cbow()
    
    
    def get_ds_ready_skip_gram(self) -> tf.data.Dataset:
        
        self.dl.prepare()
        
        list_without_stopwords = self.dl.list_of_sentences_without_stopwords_not_split()
        text_ds = tf.data.Dataset.from_tensor_slices(list_without_stopwords)
        
        self.w2i.adapt(text_ds)
        self.vocab = self.w2i.inverse_vocab
        text_vector_ds = self.w2i.vectorize(text_ds)
        
        sequences_in_int = list(text_vector_ds.as_numpy_iterator())
        sequences = [list(seq) for seq in sequences_in_int] # convert the numpy arrays in lists
        
        
        contexts, targets = self.pipeline_fn(sequences)
        
        def gen():
            yield targets.pop(), contexts.pop()
        
        #TODO : Remettre ca    
        # final_ds = tf.data.Dataset.from_generator(gen, 
        #                                             output_signature=(
        #                                             tf.TensorSpec(shape=(), dtype=tf.int64),
        #                                             tf.TensorSpec(shape=(), dtype=tf.int64)))
        
        final_ds = tf.data.Dataset.from_tensor_slices((targets, contexts))

        
        return final_ds#.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder = True).cache().prefetch(buffer_size = AUTOTUNE)
        
    
    def get_ds_ready_cbow(self) -> tf.data.Dataset:
        
        self.dl.prepare()
        
        list_without_stopwords = self.dl.list_of_sentences_without_stopwords_not_split()
        text_ds = tf.data.Dataset.from_tensor_slices(list_without_stopwords)
        
        self.w2i.adapt(text_ds)
        self.vocab = self.w2i.inverse_vocab
        text_vector_ds = self.w2i.vectorize(text_ds)
        
        sequences_in_int = list(text_vector_ds.as_numpy_iterator())
        sequences = [list(seq) for seq in sequences_in_int] # convert the numpy arrays in lists
        
        
        contexts, targets = self.pipeline_fn(sequences)
        
        def gen():
            yield contexts.pop(), targets.pop() # here it i the other way around, compared to the previous method
            
        final_ds = tf.data.Dataset.from_generator(gen, 
                                                    output_signature=(
                                                    tf.TensorSpec(shape=(), dtype=tf.int64),
                                                    tf.TensorSpec(shape=(), dtype=tf.int64)))
        
        print(final_ds)
        
        return final_ds.batch(BATCH_SIZE, drop_remainder = True).cache().prefetch(buffer_size = AUTOTUNE)
    
    def get_vocab(self):
        
        if not hasattr(self.w2i, 'inverse_vocab'):
            raise ValueError('w2i not adapted to a dataset')
        
        return self.w2i.inverse_vocab