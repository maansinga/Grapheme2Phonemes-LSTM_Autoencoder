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

# for i, (input_text, target_text) in enumerate(zip(words, phonemes)):
#     for t, char in enumerate(input_text):
#         encoder_input_data[i, t, char_2_index[char]] = 1.
#     for t, char in enumerate(['\s'] + target_text + ['\e']):
#         # decoder_output_data is ahead of decoder_input_data by one timestep
#         decoder_input_data[i, t, phoneme_2_index[char]] = 1.
#         if t > 0:
#             # decoder_output_data will be ahead by one timestep
#             # and will not include the start character.
#             decoder_output_data[i, t - 1, phoneme_2_index[char]] = 1.

# print(decoder_output_data)
# decoder_output_dd[:,:-1,:] = decoder_input_data[:, 1:, :]
# print(decoder_output_data == decoder_output_dd)
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


# Define an input sequence and process it.
encoder_lstm_input_layer = Input(
    shape=(None, word_tokens_count)
)
encoder_lstm_layer = LSTM(
    latent_dim,
    return_state=True
)

#encoder_lstm_layer input = encoder_lstm_input_layer = word tensor
encoder_lstm_outputs, encoder_lstm_state_h, encoder_lstm_state_c = encoder_lstm_layer(encoder_lstm_input_layer)

# encoder_lstm_outputs are not needed - we just need an internal representation
encoder_decoder_internal_state = [encoder_lstm_state_h, encoder_lstm_state_c]

# Decoder will start with the internal state of first lstm and incrementally build up final result
lstm2_inputs = Input(
    shape=(None, phoneme_tokens_count)
)

# This lstm will take previous lstm's generated state
decoder_lstm = LSTM(
    latent_dim, # internal representation size
    return_sequences=True, # Well, these are our phonetic sequences
    return_state=True # We need internal states of decoder - but wont use them - unless we are stacking other LSTMs
)

# Dense NN
decoder_dense = Dense(
    phoneme_tokens_count,
    activation='softmax'
)

#decoder_lstm_inputs = lstm2_inputs = decoder_input_data = phonemes tensor
# we use encoder internal state and previously predicted phoneme to
# predict the next phoneme
decoder_dense_inputs, _, _ = decoder_lstm(
    lstm2_inputs, # input of this lstm = Final results - we are forcing this lstm to map internal states to outputs
    initial_state=encoder_decoder_internal_state # initial state = internal states output of first lstm [encoder_lstm_state_h, encoder_lstm_state_c]
)
# secoder_dense_inputs = decoder_lstm_outputs
decoder_dense_outputs = decoder_dense(decoder_dense_inputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_output_data`
model = Model(
    inputs=[encoder_lstm_input_layer, lstm2_inputs], # input of lstm1 and lstm2 in a pair
    outputs=decoder_dense_outputs
)

model.summary()

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
'''
The input for model training will be 
> encoder_input_data = one-hot representations of words
> decoder_input_data = one-hot representations of phonemes

The output shall be the next phoneme based on previous words - characters and previous - phonemes
~ decoder_targer_data = decoder_input_data << 1
'''
model.fit(
    x=[encoder_input_data, decoder_input_data], #encoder input and decoder input as input for the model
    y=decoder_output_data,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2
)

# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states

# Define sampling models
'''
Now, we reassemble the model using trained variables.

encoder is a model that takes encoder_lstm_input_layer and generates encoder_decoder_internal_state
'''
encoder_model = Model(
    inputs=encoder_lstm_input_layer,
    outputs=encoder_decoder_internal_state
)

'''
Decoder has 2 inputs - the hypotheses/state and classification/output of encoder
Together, they make up decoder input.
'''
ds_input_h = Input(shape=(latent_dim,))
ds_input_c = Input(shape=(latent_dim,))

ds_inputs = [ds_input_h, ds_input_c]

# This lstm constructs internal rep from lstm2_inputs = phonemes vector
# and internal state of first lstm
decoder_dense_inputs, dh, dc = decoder_lstm(
    lstm2_inputs, initial_state=ds_inputs)
ds = [dh, dc] # internal rep

# This internal rep is passed through a Dense NN(the one we previously trained)
decoder_dense_outputs = decoder_dense(decoder_dense_inputs)

# Input to the deocder model are
# > lstm2_inputs = previous phonemes
# > H state of previous LSTM(1)
# > C state of previous LSTM(1)

# Outputs of this model are
# > Decoder dense layer's output
# > H state of this LSTM(2)
# > C state of this LSTM(2)
decoder_model = Model(
    inputs = [lstm2_inputs] + ds_inputs,
    outputs = [decoder_dense_outputs] + ds)


def decode_sequence(input_batch):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_batch)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, phoneme_tokens_count))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, phoneme_2_index['\s']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = []
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = index_2_phoneme[sampled_token_index]
        decoded_sentence += [sampled_char]

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\e' or
           len(decoded_sentence) > max_phonemes_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, phoneme_tokens_count))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


for seq_index in range(100):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_batch = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_batch)
    print('-----------------------------------------------------------------------------------------------')
    print('Input sentence:', words[seq_index])
    print('Decoded sentence:', decoded_sentence[:-1])

