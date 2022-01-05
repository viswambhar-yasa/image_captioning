import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, AveragePooling2D, Dropout, GlobalAveragePooling2D, Input, Dense, \
    Embedding, LSTM, concatenate, BatchNormalization, Bidirectional
from tensorflow.keras.applications import Xception, InceptionV3
from tensorflow.keras.models import Model, Sequential
from tensorflow.python.keras.layers.core import RepeatVector
from tensorflow.python.keras.layers.recurrent import GRU, GRUCell, LSTMCell
from tensorflow.python.training.tracking import base


def image_encoder(img_input, trainable_layers=0, CNN_Type='Xception', Embed_Size=256, disp=False):
    print('Building CNN model')
    if CNN_Type == 'Xception':
        cnn_pre_trained_model = Xception(include_top=False, weights='imagenet', input_tensor=img_input)
    else:
        cnn_pre_trained_model = InceptionV3(include_top=False, weights='imagenet', input_tensor=img_input)
    for i, layer in enumerate(cnn_pre_trained_model.layers):
        if len(cnn_pre_trained_model.layers) - i < trainable_layers:
            layer.trainable = True
        else:
            layer.trainable = False
    cnn_inputs = cnn_pre_trained_model.inputs
    base_model = cnn_pre_trained_model.output
    base_model = GlobalAveragePooling2D(name='global_average_pooling')(base_model)
    embed_image = tf.keras.layers.Dense(Embed_Size, activation='tanh', name='embed_image')(base_model)
    feature_extration_model = Model(inputs=cnn_inputs, outputs=embed_image, name='CNN encoder model')
    print('CNN model {output shape}:', embed_image.shape)
    if disp:
        tf.keras.utils.plot_model(feature_extration_model, to_file='base_model.png', show_shapes=True)
    return feature_extration_model


def txt_decoder(rnn_input, Embed_Size=256, Bi_Direction=False, RNN_Type='LSTM', RNN_Layers=1):
    print('Building RNN model')
    for i in range(RNN_Layers):
        x = BatchNormalization()(rnn_input)
        if RNN_Type == 'LSTM':
            if i == (RNN_Layers - 1):
                if Bi_Direction:
                    rnn_out = Bidirectional(LSTM(Embed_Size))(x)
                else:
                    rnn_out = LSTM(Embed_Size, return_sequences=True, return_state=True)(x)
            else:
                if Bi_Direction:
                    rnn_out = Bidirectional(LSTM(Embed_Size, return_sequences=True, return_state=True))(x)
                else:
                    rnn_out = LSTM(Embed_Size, return_sequences=True)(x)
        else:
            if i == (RNN_Layers - 1):
                if Bi_Direction:
                    rnn_out = Bidirectional(GRU(Embed_Size))(x)
                else:
                    rnn_out = GRU(Embed_Size)(x)
            else:
                if Bi_Direction:
                    rnn_out = Bidirectional(GRU(Embed_Size, return_sequences=True))(x)
                else:
                    rnn_out = GRU(Embed_Size, return_sequences=True)(x)
        rnn_input = rnn_out
    return rnn_out


def Caption_model_gen(NET, img_shape=(256, 256, 3), vocab_size=8763, Embed_Size=256, max_length=20, disp=False):
    img_input = tf.keras.Input(shape=img_shape)
    cnn_model = image_encoder(img_input, trainable_layers=0, CNN_Type='Xception', disp=False)
    embed_image = tf.keras.layers.Dense(Embed_Size, activation='tanh')(cnn_model.output)

    text_input = tf.keras.Input(shape=(max_length,))
    Embedding_layer = Embedding(input_dim=vocab_size, output_dim=Embed_Size, input_length=max_length, mask_zero=True)(
        text_input)

    whole_seq_output, final_memory_state, final_carry_state = txt_decoder(Embedding_layer, Embed_Size=Embed_Size,
                                                                          Bi_Direction=False, RNN_Type='LSTM',
                                                                          RNN_Layers=1)
    print('final_carry_state {rnn output shape}:', final_carry_state.shape)
    rnn_output = final_carry_state
    if NET == 'policy':
        image_txt_embed = tf.keras.layers.add([embed_image, rnn_output])
        print('Image and text {add shape}:', image_txt_embed.shape)
        policy_net_output = tf.keras.layers.Dense(vocab_size, activation='softmax')(image_txt_embed)
        policy_net_model = Model(inputs=[img_input, text_input], outputs=policy_net_output, name='Policy_Net')

        print('output {shape}', policy_net_output.shape)
        print('Policy Net built successfully')
        if disp:
            tf.keras.utils.plot_model(policy_net_model, to_file='policy_net.png', show_shapes=True)
        return policy_net_model
    elif NET == 'value':
        image_txt_embed = tf.keras.layers.concatenate([embed_image, rnn_output], axis=-1)
        print('Image and text {concat shape}:', image_txt_embed.shape)
        hidden_layer_1 = Dense(1024, activation='tanh', name='MLP_layer1')(image_txt_embed)
        hidden_layer_2 = Dense(512, activation='tanh', name="MLP_layer2")(hidden_layer_1)
        value_net_outputs = Dense(1, activation='tanh', name='decoder_output')(hidden_layer_2)
        value_net_model = Model(inputs=[img_input, text_input], outputs=value_net_outputs, name='Value_Net')
        print('output {shape}', value_net_outputs.shape)
        print('Value Net built successfully')
        if disp:
            tf.keras.utils.plot_model(value_net_model, to_file='value_net.png', show_shapes=True)
        return value_net_model
    else:
        feature_vector = Dense(512, activation='tanh')(embed_image)
        text_sequence_vector = Dense(512, activation='tanh', name='rnn_linear')(rnn_output)
        print('Image feature vector shape:', feature_vector.shape)
        print('Text sequence vector shape:', text_sequence_vector.shape)
        reward_model = Model(inputs=[img_input, text_input], outputs=[feature_vector, text_sequence_vector],
                             name='reward net model')
        print('Reward Net built successfully')
        if disp:
            tf.keras.utils.plot_model(reward_model, to_file='reward_net.png', show_shapes=True)
        return reward_model


if __name__ == "__main__":
    print(tf.__version__)
    actor_model = Caption_model_gen('policy')
