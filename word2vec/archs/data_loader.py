import sys
sys.path.insert(0, "../")

import tensorflow as tf
import tensorflow_datasets as tfds

try:
    from word2vec.archs.constants import CBOW_N_WORDS, SKIPGRAM_N_WORDS, MIN_WORD_FREQUENCY, MAX_SEQ_LENGTH, BATCH_SIZE, VOCAB_SIZE, BUFFER_SIZE, N_ELEMNT
except ImportError:
    raise ImportError('constants' + ' non importé')

import nltk
from nltk.corpus import stopwords
    
from typing import List, Any, Dict

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

    def __init__(self, name: str) -> None:
        """[summary]
        the datasets are always in bytes, but sometimes the lists returned are in string like combine_first_strings
        
        Args:
            path ([type]): [description]
            language ([type]): ex : 'english'
        """
        self.name = name
        
        
    def load(self) -> None:
        """
        self.ds : the dataset, can be lowered, but never split
            tf.Tensor: shape=(x,), dtype=string, where x varies
            
        self.ds_split : the dataset split
            tf.Tensor: shape=(), dtype=string
            
        """
    
        self.ds = tfds.load(name = self.name, split = 'train').take(N_ELEMNT)
        print(f'cardinality : {self.ds.cardinality()}')
        print()
        
        self.ds_split = None
        
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
    
    
    def prepare_(self, lower: bool = True, split: bool = True) -> tf.data.Dataset:
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




def pipeline_skipgram(tokenized_sequence):
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
    
    ta = tf.TensorArray(tf.int64, size = 0, dynamic_size = True, clear_after_read = False)
    i = 0
    
    
    # if len(tokenized_sequence) < SKIPGRAM_N_WORDS * 2 + 1: # we do not take this sequence into account
    #     continue
    
    if MAX_SEQ_LENGTH:
        tokenized_sequence = tokenized_sequence[:MAX_SEQ_LENGTH]
        
    msk = tf.range(start = 0, limit = 2 * SKIPGRAM_N_WORDS + 1)
    msk = tf.math.equal(msk, SKIPGRAM_N_WORDS)
    msk = tf.math.logical_not(msk)
    
    for idx in range(tokenized_sequence.shape[0])[SKIPGRAM_N_WORDS: -SKIPGRAM_N_WORDS]:
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
    
    def adapt(self, ds):
        """Takes a batched ds

        Args:
            ds ([type]): [description]
        """
        
        self.vectorize_layer.adapt(ds.batch(batch_size = BATCH_SIZE))
        self.inverse_vocab = self.vectorize_layer.get_vocabulary() # l'indice est l'entier associé, le mot le plus fréquent etant en premier
        
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
    """[summary]
    We load the data from a tf.data.Dataset
    then we must (because we can't remove stopwords directly from the tf.data.Dataset) convert it to a list of sequences
    then we split each sequences into words to remove the stopwords
    then we join each words of the same sequences to form a list of sequences without stopwords
    
    then we create a new dataset ds from this list of sequences not split without the stopwords
    we adapt the TextVectorization layer to the ds
    we vectorize ds
    
    then we turn this ds into a list of sequences in integers
    we apply the pipeline function on it
    
    we create our final ds from it, and don't batch it because it will be done in the Trainer
    """
    
    def __init__(self, type_model, name: str) -> None:
        
        assert(type_model in ['skip_gram', 'cbow'])
        self.type_model = type_model
        
        self.w2i = Word2Int()
        self.dl = DataLoader(name)
        self.pipeline_fn = pipeline_skipgram if self.type_model == 'skip_gram' else pipeline_cbow

    
    def get_ds_ready(self):
        
        return self.get_ds_ready_skip_gram() if self.type_model == 'skip_gram' else self.get_ds_ready_cbow()
    
    
    def get_ds_ready_skip_gram(self) -> tf.data.Dataset:
        
        print('.... Start preparing dataset')
        self.dl.prepare()
        print('done;')
        print()
        
        print('.... Start removing stopwords : about 1s per 1k elements of the initial cardinality of the dataset')
        list_without_stopwords = self.dl.list_of_sentences_without_stopwords_not_split()
        text_ds = tf.data.Dataset.from_tensor_slices(list_without_stopwords)
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
        
        # print('.... Converting ds as a list for pipeline')
        # sequences_in_int = list(text_vector_ds.as_numpy_iterator())
        # print('sequences_in_int list done...')
        # sequences = [list(seq) for seq in sequences_in_int] # convert the numpy arrays in lists
        # print('done;')
        # print()
        
        # print('.... Starting pipeline')
        # contexts, targets = self.pipeline_fn(sequences)
        # print('done;')
        # print()
        
        for i, e in enumerate(text_vector_ds.take(20).__iter__()):
            print(i)
            print(e)
        
        def size_filter(t):
            print(t.shape)
            print(t.shape[0])
            print(2 * SKIPGRAM_N_WORDS + 1)
            print(tf.math.greater_equal(t.shape[0], 2 * SKIPGRAM_N_WORDS + 1))
            return tf.math.greater_equal(t.shape[0], 2 * SKIPGRAM_N_WORDS + 1)
            
        print('.... Filtering 1 the dataset')
        ds = text_vector_ds.take(20).filter(size_filter)
        print('done;')
        print()    
        
        print('.... Filtering the dataset')
        ds = text_vector_ds.filter(lambda x : tf.math.greater_equal(x.shape[0], 2 * SKIPGRAM_N_WORDS + 1))
        print('done;')
        print()
        
        print('.... Mapping the pipeline function')
        ds = ds.map(self.pipeline_fn)
        print('done;')
        print()
        
        print('.... Unbatching')
        ds.unbatch()
        print('done;')
        print()
        
        # def gen():
        #     yield targets.pop(), contexts.pop()
        
        #TODO : Remettre ca    
        # final_ds = tf.data.Dataset.from_generator(gen, 
        #                                             output_signature=(
        #                                             tf.TensorSpec(shape=(), dtype=tf.int64),
        #                                             tf.TensorSpec(shape=(), dtype=tf.int64)))
        # print('.... Builing dataset')
        # final_ds = tf.data.Dataset.from_tensor_slices((targets, contexts))
        # print('done;')
        # print('')
        
        return ds#.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder = True).cache().prefetch(buffer_size = AUTOTUNE)
        
    
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