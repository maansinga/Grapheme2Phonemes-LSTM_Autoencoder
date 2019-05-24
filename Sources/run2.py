'''
@Author = Sree Teja Simha G

This is an implementation for grapheme to phoneme conversion. The lead inspiration for this
idea came from autoencoders. A specialized autoencoder for sequence to sequence mapping using RNN's
internal representation to target output.

- Sequence to Sequence Learning with Neural Networks
    https://arxiv.org/abs/1409.3215
'''
from __future__ import print_function

from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np

batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 64  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.
# Path to the data txt file on disk.
data_path = 'ph.txt'

# Vectorize the data.
words = []
phonemes = []

word_tokens = set()
phoneme_tokens = set()

with open(data_path, 'r') as source:
    lines = source.readlines()
    lines = [line[:-1] for line in lines]


for line in lines[: min(num_samples, len(lines) - 1)]:
    input_text, target_text = line.split('\t')
    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.
    target_text = target_text
    words.append(input_text)
    phonemes.append(target_text.split(' '))

word_tokens = [chr(ord('a') + i) for i in range(26)]
phoneme_tokens = [
    '\s',    'AA',    'AE',    'AH',    'AO',
    'AW',    'AY',    'B',    'CH',    'D',
    'DH',    'EH',    'ER',    'EY',    'F',
    'G',    'HH',    'IH',    'IY',    'JH',
    'K',    'L',    'M',    'N',    'NG',
    'OW',    'OY',    'P',    'R',    'S',
    'SH',    'T',    'TH',    'UH',    'UW',
    'V',    'W',    'Y',    'Z',    'ZH',  '\e'
]

word_tokens_count = len(word_tokens)
phoneme_tokens_count = len(phoneme_tokens)
max_word_length = max([len(txt) for txt in words])
max_phonemes_length = max([len(txt) for txt in phonemes]) + 2

print(phoneme_tokens)

print('Number of samples:', len(words))
print('Number of unique input tokens:', word_tokens_count)
print('Number of unique output tokens:', phoneme_tokens_count)
print('Max sequence length for inputs:', max_word_length)
print('Max sequence length for outputs:', max_phonemes_length)

char_2_index = dict([(char, i) for i, char in enumerate(word_tokens)])
phoneme_2_index = dict([(char, i) for i, char in enumerate(phoneme_tokens)])
index_2_char = dict([(i, char) for char, i in char_2_index.items()])
index_2_phoneme = dict([(i, char) for char, i in phoneme_2_index.items()])

encoder_input_data = np.zeros(
    shape=(len(words), max_word_length, word_tokens_count),
    dtype='float32'
)
decoder_input_data = np.zeros(
    shape=(len(words), max_phonemes_length, phoneme_tokens_count),
    dtype='float32'
)
decoder_output_data = np.zeros(
    shape=(len(words), max_phonemes_length, phoneme_tokens_count),
    dtype='float32'
)
decoder_output_dd = np.zeros(
    shape=(len(words), max_phonemes_length, phoneme_tokens_count),
    dtype='float32'
)


for i in range(len(words)):
    word = words[i]
    for j in range(len(word)):
        encoder_input_data[ i, j, char_2_index[word[j]] ] = 1

    word_ph = ['\s'] + phonemes[i] + ['\e']
    # print(word_ph)
    for k in range(len(word_ph)):
        word_phk = word_ph[k]
        # print(word_phk)
        decoder_input_data[ i, k, phoneme_2_index[word_phk] ] = 1

decoder_output_data[:, :-1, :] = decoder_input_data[:, 1:, :]


internal_representation_size = 128
input_layer = Input(shape=(None, word_tokens_count), name='input_layer')
lstm1_layer = LSTM(internal_representation_size, return_state=True, return_sequences=True, name='lstm1_layer')
lstm2_layer = LSTM(internal_representation_size, return_state=True, return_sequences=True, name='lstm2_layer')