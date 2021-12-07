CBOW_N_WORDS = 3
SKIPGRAM_N_WORDS = 3 # contexte = 3 mots avant, 3 mots après le mot target

MIN_WORD_FREQUENCY = 30
MAX_SEQ_LENGTH = 256 # à definir à zéro si on ne veut pas de troncature des paragraphes

EMBEDDING_DIM = 256

VOCAB_SIZE = 4096
BATCH_SIZE = 32

LANGUAGE = 'english'

BUFFER_SIZE = 8192

PATH = '/coding_linux20/programming/datasets/wikitext-2-raw' + '/wiki.train.raw'