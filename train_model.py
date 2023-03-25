import pandas as pd
import numpy as np
import re
import tensorflow as tf
from tensorflow.keras import Sequential, Input, Model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint


# PREPARE DATAFRAME WITH DATA
print("Beginning dataframe preparation...")
def clean_text(df, col):
    df[col] = df[col].str.lower().replace("[^a-zA-Z]+", "")
    df[col] = df[col].str.rstrip()
    return df

df = pd.read_csv('data/lyrics-data.csv')
df = df[df['Idiom'] == 'ENGLISH']
df = pd.concat([df['SName'], df['Lyric']], axis=1, keys=['title', 'lyrics'])

for col in ['title', 'lyrics']:
    df = clean_text(df, col)
print("Dataframe preparation complete")


# PREPARE DATASET METADATA FOR TRAINING
print("Prepare dataset metadata and indices...")
words = ""
for song in df['lyrics']:
    words += song
for title in df['title']:
    words += title

word_set = set(words)

char2idx = {'<BOS>':0, '<EOS>':1}
counter = 2
for word in word_set:
    char2idx[word] = counter
    counter += 1
idx2char = {}
for word, idx in char2idx.items():
    idx2char[idx] = word

vocab_size = len(char2idx)
print("Metadata and indices successfully generated")


# PREPARE TRAINING DATA
print("Preparing training data...")
max_input_len = 75
max_output_len = 50
data_size = 20000 #len(df['lyrics'])

onehot_encoder_input = np.zeros(shape=(data_size, max_input_len, vocab_size))
for i, song in enumerate(df['lyrics']):
    if i >= data_size:
        break
    for j, ch in enumerate(song):
        if j >= max_input_len:
            break
        charidx = char2idx[ch]
        onehot_encoder_input[i][j][charidx] = 1

onehot_decoder_input = np.zeros(shape=(data_size, max_output_len, vocab_size))
for i in range(onehot_decoder_input.shape[0]):
    onehot_decoder_input[i][0][char2idx["<BOS>"]] = 1

for i, title in enumerate(df['title']):
    if i >= data_size:
        break
    for j, ch in enumerate(title[:-1]):
        if j+1 >= max_output_len:
            break
        charidx = char2idx[ch]
        onehot_decoder_input[i][j+1][charidx] = 1

onehot_target = np.zeros(shape=(data_size, max_output_len, vocab_size))
for i, title in enumerate(df['title']):
    if i >= data_size:
        break
    for j, ch in enumerate(title):
        if j >= max_output_len:
            break
        charidx = char2idx[ch]
        onehot_target[i][j][charidx] = 1
print("Training data successfully prepared")


# PREPARE MODEL
print("Preparing model...")
latent_dim = 32

def get_model(max_input_len, max_output_len, vocab_size):
    encoder_inputs = Input(shape=(max_input_len, vocab_size))
    encoder = LSTM(latent_dim, input_shape=(None, max_input_len, vocab_size), return_state=True)
    _, state_h, state_c = encoder(encoder_inputs)  # discard encoder sequence

    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(max_output_len, vocab_size))

    decoder = LSTM(latent_dim, input_shape=(None, max_output_len, vocab_size), return_state=True, return_sequences=True)
    decoder_outputs, _, _ = decoder(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model


def get_encoder_decoder(model):
    # Build encoder
    encoder_inputs = model.input[0]
    encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output  # lstm_1
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = Model(encoder_inputs, encoder_states)

    # Build decoder
    decoder_inputs = model.input[1]
    decoder_state_input_h = Input(shape=(latent_dim,), name="input_3")
    decoder_state_input_c = Input(shape=(latent_dim,), name="input_4")
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_lstm = model.layers[3]
    decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs
    )
    decoder_states = [state_h_dec, state_c_dec]
    decoder_dense = model.layers[4]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
    )

    return encoder_model, decoder_model
model = get_model(max_input_len, max_output_len, vocab_size)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics='accuracy')
model.summary()
print("Model successfully prepared")


# TRAIN MODEL
print("Training model starting...")
batch_size = 32
epochs = 1

model_checkpoint_callback = ModelCheckpoint(
            filepath="./chkpt",
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=False,
            save_freq=batch_size*2)


history = model.fit(
        [onehot_encoder_input, onehot_decoder_input],
        onehot_target,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
        callbacks=[model_checkpoint_callback]
    )
model.save("model_save_pyfile")
print("Model training finished")


# GENERATE PREDICTIONS FROM MODEL
print("Prediction generation beginning")
def predict(model, input_text, vocab_size, max_input_len):
    input_text = input_text.lower().split(" ")

    encoder_model, decoder_model = get_encoder_decoder(model)

    input_seq = np.zeros((1, max_input_len, vocab_size))
    
    for i, ch in enumerate(input_text):
        if i >= max_input_len:
            break
        if ch in char2idx:
            input_seq[0][i][char2idx[ch]] = 1.0
    
    if len(input_text) < max_input_len:
        input_seq[0][len(input_text)][char2idx["<EOS>"]] = 1.0


    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)
    
    #print(len(states_value))
    print(states_value)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, max_output_len, vocab_size))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, char2idx["<BOS>"]] = 1.0
    
    #print(target_seq[0,0,:])

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ""
    counter = 1
    while not stop_condition:
        print([target_seq] + states_value)
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample 4 best tokens and pick one randomly
        sampled_token_idxs = np.argpartition(output_tokens[0,-1,:], -4)[-4:]
        #print(sampled_token_idxs)
        #print(output_tokens[0,-1,sampled_token_idxs])
        rand_idx = np.random.randint(0,len(sampled_token_idxs))
        sampled_token_index = sampled_token_idxs[rand_idx]
        sampled_char = idx2char[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if sampled_char == "<EOS>" or len(decoded_sentence) >= max_output_len:
            stop_condition = True
            break
        
        print(f"counter = {counter}, len(decoded_sentence) = {len(decoded_sentence)}, max_output_len = {max_output_len}")
        # Update the target sequence (of length 1).
        #target_seq = np.zeros((1, max_output_len, vocab_size))
        target_seq[0, counter, sampled_token_index] = 1.0
        counter += 1

        # Update states
        states_value = [h, c]
    print(target_seq)
    print(target_seq[0,1,:])
    return decoded_sentence


test_sentence = "Would the real slim shady please stand up"
predict(model, test_sentence, vocab_size, max_input_len)
print("Prediction generation done")