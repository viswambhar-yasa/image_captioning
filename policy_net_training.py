import tensorflow as tf
from main import Caption_model_gen
from data_processing import data_processing
from data_generator import captions_generation
import pickle

print('TensorFlow Version', tf.__version__)
vocab_size = 5000
max_length = 25




captions_text_path = r'.\Flicker8k_Dataset\text_files\Flickr8k.token.txt'
captions_extraction = data_processing(captions_text_path)
trn_images_id_text = r'.\Flicker8k_Dataset\text_files\Flickr_8k.trainImages.txt'
train_cleaned_seq, train_cleaned_dic = captions_extraction.cleaning_sequencing_captions(trn_images_id_text)
val_images_id_text = r'.\Flicker8k_Dataset\text_files\Flickr_8k.devImages.txt'
val_cleaned_seq, val_cleaned_dic = captions_extraction.cleaning_sequencing_captions(val_images_id_text)
test_images_id_text = r'.\Flicker8k_Dataset\text_files\Flickr_8k.testImages.txt'
test_cleaned_seq, test_cleaned_dic = captions_extraction.cleaning_sequencing_captions(test_images_id_text)
captions_extraction.tokenization(train_cleaned_seq, vocab_size)
print("No of captions: Training-" + str(len(train_cleaned_seq) / 5) + " Validation-" + str(
    len(val_cleaned_seq) / 5) + " test-" + str(len(test_cleaned_seq) / 5))

train_cap_tok = captions_extraction.sentence_tokenizing(train_cleaned_dic)
val_cap_tok = captions_extraction.sentence_tokenizing(val_cleaned_dic)
test_cap_tok = captions_extraction.sentence_tokenizing(test_cleaned_dic)

image_pth_rt = r".\Flicker8k_Dataset" + r"\\"
trn_dataset = captions_generation(train_cap_tok, vocab_size, image_pth_rt, max_length)
val_dataset = captions_generation(val_cap_tok, vocab_size, image_pth_rt, max_length)

inputs, outputs = next(iter(trn_dataset))
print(inputs[0].shape, inputs[1].shape, outputs.shape)

actor_model = Caption_model_gen(NET='policy', vocab_size=vocab_size, Embed_Size=512, max_length=max_length-1)
actor_model.summary()
actor_model.compile(loss=tf.keras.losses.categorical_crossentropy,
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    metrics=['accuracy'])


def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

checkpoint_filepath = '/content'
early_stop_callback = tf.keras.callbacks.EarlyStopping(
    monitor='loss', patience=10)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='accuracy',
    mode='auto')

callback = [early_stop_callback, model_checkpoint_callback]

history = actor_model.fit(trn_dataset, steps_per_epoch=10, epochs=60, shuffle=False, validation_data=val_dataset,
                          validation_steps=5)

model_parameters = history.history

f = open("/content/history_model_0.001.pkl", "wb")
pickle.dump(model_parameters, f)
f.close()
