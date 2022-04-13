## Image captioning using reinforcement learning
### Policy to actor method on deep convolution and recurrent networks
#### Project Seminar for artifical intelligence WS2021-22
##### Authors : Viswambhar Yasa, Venkata Mukund
# ## This file contains loss function which are required to build policy, value and reward net
import tensorflow as tf
def reward_net_loss(visual_features,sematic_featurres,margin=0.4):
    """
    Visual sematic embedding 
    Args:
        visual_features (numpy array): feature vector of an image
        sematic_featurres (_type_): feature vector of the caption
        margin (float, optional): A constant used for building the embeded value. Defaults to 0.4.
    # From visual-sematic embedding based on equ. 3.2
    Returns:
        _type_: loss value
    """
    # shape of the feature vector
    B=visual_features.shape[0]
    remove_diagonal_matrix=tf.ones(B)-tf.eye(B)
    similarity_score_1=tf.einsum("ij,kj->ik",visual_features,sematic_featurres)
    similarity_score_2=tf.einsum("ij,kj->ki",visual_features,sematic_featurres)
    diagonal=tf.einsum("ii->i",similarity_score_1)
    s_ii=tf.eye(B)*tf.transpose(diagonal)
    loss1=tf.reduce_sum(tf.maximum(0,margin+(similarity_score_1*remove_diagonal_matrix)-s_ii))
    loss2=tf.reduce_sum(tf.maximum(0,margin+(similarity_score_2*remove_diagonal_matrix)-s_ii))
    rn_loss=(loss1+loss2)*(1/B)
    return rn_loss

def loss(image_encoder,caption_encoder):
    """
    Visual sematic embedding 
    Args:
        image_encoder (numpy array): feature vector of an image
        caption_encoder (array): feature vector of the caption
    # From visual-sematic embedding based on equ. 3.4
    Returns:
        _type_: loss value
    """
    gamma=0.2
    # shape of the feature vector
    N,D=image_encoder.shape
    img_encode = image_encoder
    cap_encode = caption_encoder
    scores_matrix = tf.matmul(img_encode,tf.transpose(cap_encode))
    diagonal = tf.linalg.diag_part(scores_matrix)
    cost_cap = tf.maximum(0.0, gamma - diagonal + scores_matrix)
    diagonal = tf.reshape(diagonal, [-1, 1])
    cost_img = tf.maximum(0.0, gamma - diagonal + scores_matrix)
    cost_cap = tf.linalg.set_diag(cost_cap, [0]*N)
    cost_img = tf.linalg.set_diag(cost_img, [0]*N)
    loss = tf.reduce_sum(cost_img) + tf.reduce_sum(cost_cap)
    return loss/N

def cos(img_encode, cap_encode):
    """ cosine distance between two feature vector

    Args:
        img_encode (array): feature vector of an image
        cap_encode (array): feature vector of the caption

    Returns:
        _type_: cosine distance
    """
    inner_product = tf.reduce_sum(tf.multiply(img_encode, cap_encode), axis=1)
    norm1 = tf.sqrt(tf.reduce_sum(tf.square(img_encode), axis=1))
    norm2 = tf.sqrt(tf.reduce_sum(tf.square(cap_encode), axis=1))
    cos = inner_product / (norm1 * norm2)
    return cos

def Rewards(reward_model,input):
    """ Calculating the reward value (for evaluation with value)

    Args:
        reward_model (): reward model
        input (array): caption of the image

    Returns:
        _type_: _description_
    """
    # extracting visual and sematic feature vectore from reward net
    visEmbeds, semEmbeds = reward_model(input)
    inner_product = tf.reduce_sum(tf.multiply(visEmbeds, semEmbeds), axis=1)
    norm1 = tf.sqrt(tf.reduce_sum(tf.square(visEmbeds), axis=1))
    norm2 = tf.sqrt(tf.reduce_sum(tf.square(semEmbeds), axis=1))
    # calulating the cosine distance between visual and sematic feature vector
    cos = inner_product / (norm1 * norm2)
    return cos