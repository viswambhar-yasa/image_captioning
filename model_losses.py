import tensorflow as tf
def reward_net_loss(visual_features,sematic_featurres,margin=0.4):
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
    gamma=0.2
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
    inner_product = tf.reduce_sum(tf.multiply(img_encode, cap_encode), axis=1)
    norm1 = tf.sqrt(tf.reduce_sum(tf.square(img_encode), axis=1))
    norm2 = tf.sqrt(tf.reduce_sum(tf.square(cap_encode), axis=1))
    cos = inner_product / (norm1 * norm2)
    return cos

def Rewards(reward_model,input):
    visEmbeds, semEmbeds = reward_model(input)
    inner_product = tf.reduce_sum(tf.multiply(visEmbeds, semEmbeds), axis=1)
    norm1 = tf.sqrt(tf.reduce_sum(tf.square(visEmbeds), axis=1))
    norm2 = tf.sqrt(tf.reduce_sum(tf.square(semEmbeds), axis=1))
    cos = inner_product / (norm1 * norm2)
    return cos