## Image captioning using reinforcement learning
### Policy to actor method on deep convolution and recurrent networks
#### Project Seminar for artifical intelligence WS2021-22
##### Authors : Viswambhar Yasa, Venkata Mukund
## This file contains data preprocessing functions which performes cleaning and tokenization

import tensorflow as tf
class data_processing:
    def __init__(self, text_file_path):
        """
        Data preprocessing : performs data processing like extraction, cleaning and tokenization
        Args:
            text_file_path (str): path of the annoatation file
        """
        self.text_file_path = text_file_path
        self.tokenizer = None

    def extraction_captions(self, images_id_text):
        """
        Extracts the annoatation text file for processing by converting it into dictionary 
        Args:
            images_id_text (str): list of all images names

        Returns:
            _type_: _description_
        """
        # dictionary to contain captions and images
        description_map = dict()
        text = open(self.text_file_path, 'r', encoding='utf-8').read()
        images = open(images_id_text, 'r', encoding='utf-8').read()
        img_dic = []
        # images name persent in the file
        for img_id in images.split('\n'):
            img_dic.append(img_id)
        # splittng annotation 
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
        """
        Cleaning the sequence by removing all special characters, adding start and end sequence and symbols
        Args:
            images_id_text (str): annotations

        Returns:
            _type_: _description_
        """
        captions_dic = self.extraction_captions(images_id_text)
        caption_list = []
        # looping over dictionary extracting image id and captions 
        for img_id, des_list in captions_dic.items():
            for i in range(len(des_list)):
                caption = des_list[i]
                caption = ''.join(caption)
                caption = caption.split(' ')
                caption = [word.lower() for word in caption if len(word) > 1 and word.isalpha()]
                caption = ' '.join(caption)
                # adding start and end sequence
                des_list[i] = 'startseq ' + caption + ' endseq'
                caption_list.append('startseq ' + caption + ' endseq')
        max_length = max(len(des.split()) for des in caption_list)
        print('max_length of captions', max_length)
        return caption_list,captions_dic

    def tokenization(self, captions_for_token, num_wrds=5000) -> None:
        """
        generates tokenization 

        Args:
            captions_for_token (str): list of captions
            num_wrds (int, optional): Number of words in volcabulary. Defaults to 5000.
        """
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=num_wrds, oov_token='<unknw>')
        tokenizer.fit_on_texts(captions_for_token)
        self.tokenizer = tokenizer
        pass

    def sentence_tokenizing(self, captions_dic) -> dict:
        """
        Performs tokenization (converts words to numbers)

        Args:
            captions_dic (dic): annotations with images 

        Returns:
            dict: _description_
        """
        token_cap_dic = dict()
        print('Vocab size', self.tokenizer.num_words)
        for img_id, des_list in captions_dic.items():
            for i in range(len(des_list)):
                caption = des_list[i]
                # converting sentences to tokenized sentence
                cap_token = self.tokenizer.texts_to_sequences([str(caption)])
                if img_id not in token_cap_dic:
                    token_cap_dic[img_id] = list()
                token_cap_dic[img_id].append(cap_token)
        return token_cap_dic