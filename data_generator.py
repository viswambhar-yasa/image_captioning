import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.inception_v3 import InceptionV3, preprocess_input
import numpy as np
from data_processing import data_processing


def load_preprocess_img(img_path):
    img = load_img(img_path, target_size=(256, 256, 3))
    x = img_to_array(img)
    x /= 255.0
    return x


def captions_generation_reward(captions_dic, vocab_size, image_pth_rt, max_length=25, num_photos_per_batch=5, num_captions=1):
    images, input_text_seq = list(), list()
    batch_iter = 0
    batch_keys = []
    while True:
        for key, desc_list in captions_dic.items():
            # print(key)
            batch_keys.append(key)
            batch_iter += 1
            caption = 0
            # retrieve the photo feature
            photo = load_preprocess_img(image_pth_rt + key)
            for desc in desc_list:
                caption += 1
                desc = np.squeeze(desc)
                input_sequence = []
                input_seq = tf.keras.preprocessing.sequence.pad_sequences([desc], maxlen=max_length,
                                                                          padding='post')
                input_text_seq.append(input_seq)
                images.append(photo)
                if caption == num_captions:
                    break
            if batch_iter == num_photos_per_batch:
                input_text_seq = np.concatenate(input_text_seq)
                yield [np.array(images), np.array(input_text_seq)]
                images, input_text_seq = list(), list()
                batch_iter = 0
if __name__ == "__main__":
    print('TensorFlow Version', tf.__version__)
    captions_text_path = r'.\Flicker8k_Dataset\text_files\Flickr8k.token.txt'
    captions_extraction = data_processing(captions_text_path)
    trn_images_id_text = r'.\Flicker8k_Dataset\text_files\Flickr_8k.trainImages.txt'
    train_cleaned_seq,train_cleaned_dic = captions_extraction.cleaning_sequencing_captions(trn_images_id_text)
    val_images_id_text = r'.\Flicker8k_Dataset\text_files\Flickr_8k.devImages.txt'
    val_cleaned_seq,val_cleaned_dic = captions_extraction.cleaning_sequencing_captions(val_images_id_text)
    test_images_id_text = r'.\Flicker8k_Dataset\text_files\Flickr_8k.testImages.txt'
    test_cleaned_seq,test_cleaned_dic = captions_extraction.cleaning_sequencing_captions(test_images_id_text)
    captions_extraction.tokenization(train_cleaned_seq, num_wrds=5000)
    print("No of captions: Training-" + str(len(train_cleaned_seq) / 5) + " Validation-" + str(
        len(val_cleaned_seq) / 5) + " test-" + str(len(test_cleaned_seq) / 5))

    train_cap_tok = captions_extraction.sentence_tokenizing(train_cleaned_dic)
    val_cap_tok = captions_extraction.sentence_tokenizing(val_cleaned_dic)
    test_cap_tok = captions_extraction.sentence_tokenizing(test_cleaned_dic)

    image_pth_rt = r".\Flicker8k_Dataset"+r"\\"
    trn_dataset = captions_generation(train_cap_tok, 5000, image_pth_rt)
    inputs, outputs = next(iter(trn_dataset))
    print(inputs[0].shape, inputs[1].shape, outputs.shape)
