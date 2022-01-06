import tensorflow as tf


class data_processing:
    def __init__(self, text_file_path):
        self.text_file_path = text_file_path
        self.tokenizer = None

    def extraction_captions(self, images_id_text):
        description_map = dict()
        text = open(self.text_file_path, 'r', encoding='utf-8').read()
        images = open(images_id_text, 'r', encoding='utf-8').read()
        img_dic = []
        for img_id in images.split('\n'):
            img_dic.append(img_id)
        for lines in text.split('\n'):
            line_split = lines.split('\t')
            if line_split == ['']:
                continue
            image_id = line_split[0][:-2]
            image_des = line_split[1]
            if image_id in img_dic:
                if image_id not in description_map:
                    description_map[image_id] = list()
                description_map[image_id].append(image_des)
        return description_map

    def cleaning_sequencing_captions(self, images_id_text):
        captions_dic = self.extraction_captions(images_id_text)
        caption_list = []
        for img_id, des_list in captions_dic.items():
            for i in range(len(des_list)):
                caption = des_list[i]
                caption = ''.join(caption)
                caption = caption.split(' ')
                caption = [word.lower() for word in caption if len(word) > 1 and word.isalpha()]
                caption = ' '.join(caption)
                des_list[i] = 'startseq ' + caption + ' endseq'
                caption_list.append('startseq ' + caption + ' endseq')
        max_length = max(len(des.split()) for des in caption_list)
        print('max_length of captions', max_length)
        return caption_list,captions_dic

    def tokenization(self, captions_for_token, num_wrds=5000) -> None:
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=num_wrds, oov_token='<unknw>')
        tokenizer.fit_on_texts(captions_for_token)
        self.tokenizer = tokenizer
        pass

    def sentence_tokenizing(self, captions_dic) -> dict:
        token_cap_dic = dict()
        print('Vocab size', self.tokenizer.num_words)
        for img_id, des_list in captions_dic.items():
            for i in range(len(des_list)):
                caption = des_list[i]
                cap_token = self.tokenizer.texts_to_sequences([str(caption)])
                if img_id not in token_cap_dic:
                    token_cap_dic[img_id] = list()
                token_cap_dic[img_id].append(cap_token)
        return token_cap_dic


if __name__ == "__main__":
    print('TensorFlow Version', tf.__version__)
    captions_text_path = r'.\Flicker8k_Dataset\text_files\Flickr8k.token.txt'
    captions_extraction = data_processing(captions_text_path)
    trn_images_id_text = r'.\Flicker8k_Dataset\text_files\Flickr_8k.trainImages.txt'
    train_cleaned_seq = captions_extraction.cleaning_sequencing_captions(trn_images_id_text)
    val_images_id_text = r'.\Flicker8k_Dataset\text_files\Flickr_8k.devImages.txt'
    val_cleaned_seq = captions_extraction.cleaning_sequencing_captions(val_images_id_text)
    test_images_id_text = r'.\Flicker8k_Dataset\text_files\Flickr_8k.testImages.txt'
    test_cleaned_seq = captions_extraction.cleaning_sequencing_captions(test_images_id_text)
    captions_extraction.tokenization(train_cleaned_seq, num_wrds=5000)
    print("No of captions: Training-"+str(len(train_cleaned_seq)/5)+" Validation-"+str(len(val_cleaned_seq)/5)+" test"+str(len(test_cleaned_seq)/5))

