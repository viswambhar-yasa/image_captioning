## Image captioning using reinforcement learning
### Policy to actor method on deep convolution and recurrent networks
#### Project Seminar for artifical intelligence WS2021-22
##### Authors : Viswambhar Yasa, Venkata Mukund
# ## This file contains enocder and decoder function which are required to build policy, value and reward net
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Embedding, LSTM, BatchNormalization, Bidirectional
from tensorflow.keras.applications import Xception, InceptionV3,ResNet50
from tensorflow.keras.models import Model
from tensorflow.python.keras.layers.recurrent import GRU

def image_encoder(img_input, trainable_layers=0, CNN_Type='Xception', Embed_Size=512, display=False):
    """
    CNN encoder : creates a convolution layer based model
    Args:
        img_input (tuple): shape of the image
        trainable_layers (int, optional): Creates a trainable layer in base model. Defaults to 0.
        CNN_Type (str, optional): To select the base model. Defaults to 'Xception'.
        Embed_Size (int, optional): The size of the feature vector. Defaults to 512.
        display (bool, optional): displays the summary. Defaults to False.

    Returns:
        _type_: _description_
    """
    print('Building CNN model')
    # different type of base model
    if CNN_Type == 'Xception':
        cnn_pre_trained_model = Xception(include_top=False, weights='imagenet', input_tensor=img_input)
    elif CNN_Type =='ResNet':
      cnn_pre_trained_model =ResNet50(include_top=False,weights='imagenet',input_tensor=img_input,pooling='avg')
    else:
        cnn_pre_trained_model = InceptionV3(include_top=False, weights='imagenet', input_tensor=img_input)
    # creates trainable layers 
    for i, layer in enumerate(cnn_pre_trained_model.layers):
        if len(cnn_pre_trained_model.layers) - i < trainable_layers:
            layer.trainable = True
        else:
            layer.trainable = False
    cnn_inputs = cnn_pre_trained_model.inputs
    base_model = cnn_pre_trained_model.output
    # feature vector size based on base model
    if CNN_Type=='ResNet':
      embed_image = tf.keras.layers.Dense(Embed_Size, activation='relu', name='embed_image')(base_model)
    else:
      embed_image = GlobalAveragePooling2D(name='global_average_pooling')(base_model)
    feature_extraction_model = Model(inputs=cnn_inputs, outputs=embed_image, name='CNN encoder model')
    print('CNN model {output shape}:', embed_image.shape)
    if display:
        tf.keras.utils.plot_model(feature_extraction_model, to_file='base_model.png', show_shapes=True)
    return feature_extraction_model


def txt_decoder(rnn_input, Embed_Size=256, Bi_Direction=False, RNN_Type='LSTM', RNN_Layers=2):
    """
    Decoder : creates a recurrent layer 
    Args:
        rnn_input (int): size of the recurrent layer
        Embed_Size (int, optional): size of the embedding layer. Defaults to 256.
        Bi_Direction (bool, optional): Bi-directional layer options. Defaults to False.
        RNN_Type (str, optional): Type of recurrent neural network. Defaults to 'LSTM'.
        RNN_Layers (int, optional): Number of recurrent neural network. Defaults to 2.

    Returns:
        _type_: _description_
    """
    print('Building RNN model')
    for i in range(RNN_Layers):
        x = BatchNormalization()(rnn_input)
        if RNN_Type == 'LSTM':
            if i == (RNN_Layers - 1):
                if Bi_Direction:
                    rnn_out = Bidirectional(LSTM(int(Embed_Size/2)))(x)
                else:
                    rnn_out = LSTM(Embed_Size)(x)
            else:
                if Bi_Direction:
                    rnn_out = Bidirectional(LSTM(int(Embed_Size/2), return_sequences=True))(x)
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
                    rnn_out = Bidirectional(GRU(Embed_Size/2, return_sequences=True))(x)
                else:
                    rnn_out = GRU(Embed_Size, return_sequences=True)(x)
        rnn_input = rnn_out
    return rnn_out


def Caption_model_gen(NET,Embedding_layer=None, img_shape=(256, 256, 3), vocab_size=5000, Embed_Size=256, max_length=20, display=False):
    """
    created encoder-decoder based model 

    Args:
        NET (_type_): Type of net (policy,value, reward)
        Embedding_layer (_type_, optional): size of the embedding layer. Defaults to None.
        img_shape (tuple, optional): shape of the input image. Defaults to (256, 256, 3).
        vocab_size (int, optional): size of the vocabulary . Defaults to 5000.
        Embed_Size (int, optional): _description_. Defaults to 256.
        max_length (int, optional): max length of the captions. Defaults to 20.
        display (bool, optional): to plot archiecture of the model. Defaults to False.

    Returns:
        _type_: _description_
    """
    img_input = tf.keras.Input(shape=img_shape)
    cnn_model = image_encoder(img_input, trainable_layers=0, CNN_Type='Xception', display=False)
    embed_image = tf.keras.layers.Dense(Embed_Size, activation='tanh')(cnn_model.output)

    text_input = tf.keras.Input(shape=(max_length,))
    if Embedding_layer==None:
      Embedding_layer = Embedding(input_dim=vocab_size, output_dim=Embed_Size, input_length=max_length, mask_zero=True)(
        text_input)
    else:
      Embedding_layer=Embedding_layer(text_input)

    whole_seq_output = txt_decoder(Embedding_layer, Embed_Size=Embed_Size,
                                                                          Bi_Direction=False, RNN_Type='LSTM',
                                                                          RNN_Layers=1)
    print('final_carry_state {rnn output shape}:', whole_seq_output.shape)
    rnn_output = whole_seq_output
    if NET == 'policy':
        image_txt_embed = tf.keras.layers.add([embed_image, rnn_output])
        print('Image and text {add shape}:', image_txt_embed.shape)
        policy_net_output = tf.keras.layers.Dense(vocab_size, activation='softmax')(image_txt_embed)
        policy_net_model = Model(inputs=[img_input, text_input], outputs=policy_net_output, name='Policy_Net')

        print('output {shape}', policy_net_output.shape)
        print('Policy Net built successfully \n')
        if display:
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
        print('Value Net built successfully \n')
        if display:
            tf.keras.utils.plot_model(value_net_model, to_file='value_net.png', show_shapes=True)
        return value_net_model
    else:
        feature_vector = Dense(512, activation='tanh')(embed_image)
        text_sequence_vector = Dense(512, activation='tanh', name='rnn_linear')(rnn_output)
        print('Image feature vector shape:', feature_vector.shape)
        print('Text sequence vector shape:', text_sequence_vector.shape)
        reward_model = Model(inputs=[img_input, text_input], outputs=[feature_vector, text_sequence_vector],
                             name='reward net model')
        print('Reward Net built successfully \n')
        if display:
            tf.keras.utils.plot_model(reward_model, to_file='reward_net.png', show_shapes=True)
        return reward_model


if __name__ == "__main__":
    print('TensorFlow Version', tf.__version__)
    #actor_model = Caption_model_gen('policy')
    #critic_model = Caption_model_gen('value')
    #reward = Caption_model_gen('reward')
