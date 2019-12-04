#!/usr/bin/env python
# coding: utf-8

# # Обучение модели 

# In[1]:


import tensorflow as tf
import numpy as np
import os
# from utils import read_wav, extract_feats, read_dataset, batch, decode
from IPython.display import Audio
from sklearn.model_selection import train_test_split
from IPython.core.display import HTML
from scipy.signal import spectrogram




# In[2]:


import scipy.io.wavfile as wav
from python_speech_features import fbank, mfcc
import time

#from itertools import izip_longest as zip_longest

from keras.layers import LSTM, Dense, Convolution1D
from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed, Bidirectional
from itertools import zip_longest

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


def decode(d, mapping):
    """Decode."""
    shape = d.dense_shape
    batch_size = shape[0]
    ans = np.zeros(shape=shape, dtype=int)
    seq_lengths = np.zeros(shape=(batch_size, ), dtype=np.int)
    for ind, val in zip(d.indices, d.values):
        ans[ind[0], ind[1]] = val
        seq_lengths[ind[0]] = max(seq_lengths[ind[0]], ind[1] + 1)
    ret = []
    for i in range(batch_size):
        ret.append("".join(map(lambda s: mapping[s], ans[i, :seq_lengths[i]])))
    return ret


# In[4]:


def list_2d_to_sparse(list_of_lists):
    """Convert python list of lists to a [tf.SparseTensorValue](https://www.tensorflow.org/api_docs/python/tf/SparseTensorValue).

    Args:
        list_of_lists: list of lists to convert.

    Returns:
        tf.SparseTensorValue which is a namedtuple (indices, values, shape) where:

            * indices is a 2-d numpy array with shape (sum_all, 2) where sum_all is a
            sum over i of len(l[i])

            * values is a 1-d numpy array with shape (sum_all, )
0
            * shape = np.array([len(l), max_all]) where max_all is a max over i of
            len(l[i])

        Also, the following is true: for all i values[i] ==
        list_of_lists[indices[i][0]][indices[i][1]]

    """
    indices, values = [], []
    for i, sublist in enumerate(list_of_lists):
        for j, value in enumerate(sublist):
            indices.append([i, j])
            values.append(value)
    dense_shape = [len(list_of_lists), max(map(len, list_of_lists))]
    return tf.SparseTensorValue(indices=np.array(indices),
                                values=np.array(values),
                                dense_shape=np.array(dense_shape))


# In[5]:


vocabulary = { 'а': 1,
               'б': 2,
               'в': 3,
               'г': 4,
               'д': 5,
               'е': 6,
               'ё': 7,
               'ж': 8,
               'з': 9,
               'и': 10,
               'й': 11,
               'к': 12,
               'л': 13,
               'м': 14,
               'н': 15,
               'о': 16,
               'п': 17,
               'р': 18,
               'с': 19,
               'т': 20,
               'у': 21,
               'ф': 22,
               'х': 23,
               'ц': 24,
               'ч': 25,
               'ш': 26,
               'щ': 27,
               'ъ': 28,
               'ы': 29,
               'ь': 30,
               'э': 31,
               'ю': 32,
               'я': 33}

inv_mapping = dict(zip(vocabulary.values(), vocabulary.keys()))
inv_mapping[34]='<пробел>'


# In[6]:


# Каталог с базой аудио
voxforge_dir = './Voxforge'


# In[7]:


speaker_folders = os.listdir(voxforge_dir)


# In[8]:


speaker_folders1=[]
x=0
for i in speaker_folders:
    speaker_folders1.append(i)
    x=x+1
    if x>75:
        break


# In[9]:


speaker_folders1[1]


# In[10]:


def list_files_for_speaker(speaker, folder):
    """
    Generates a list of wav files for a given speaker from the voxforge dataset.
    Args:
        speaker: substring contained in the speaker's folder name, e.g. 'Aaron'
        folder: base folder containing the downloaded voxforge data

    Returns: list of paths to the wavfiles
    """

    speaker_folders = [d for d in os.listdir(folder) if speaker in d]
    wav_files = []

    for d in speaker_folders:
        for f in os.listdir(os.path.join(folder, d, 'wav')):
            wav_files.append(os.path.abspath(os.path.join(folder, d, 'wav', f)))

    return wav_files


# In[11]:


def extract_features_and_targets(wav_file):
    #wav=[]
#     обработка текст
    txt_file = wav_file.replace('/wav/', '/txt/').replace('.wav', '.txt')
    try:
        f = open(txt_file, 'rb')
    except IOError as e:
#         print(u'не удалось открыть файл')
        return 0,0,0,0
    else:
        with f:
#     with open(txt_file, 'rb') as f:
            for line in f.readlines():
                if line[0] == ';':
                    continue

            # Get only the words between [a-z] and replace period for none
            
            original = ' '.join(str(line,'utf-8').strip().lower().split(' ')).replace('.', '').replace("'", '').replace('-', '').replace(',','')
            targets = original.replace(' ', '  ')
            targets = targets.split(' ')
            stroka=[34]

            for i in targets:
                i1=i.encode("UTF-8")
                for j in range(0,len(i1),2):
                    stroka.append(vocabulary.get(i1[j:j+2].decode("utf-8"),34))
                if stroka[-1] != 34:
                    stroka.append(34)
#                     обработка звука
            fs, audio = wav.read(wav_file)

            features = mfcc(audio, samplerate=fs, lowfreq=50)

            mean_scale = np.mean(features, axis=0)
            std_scale = np.std(features, axis=0)

            features = (features - mean_scale[np.newaxis, :]) / std_scale[np.newaxis, :]

            seq_len = features.shape[0]
            return features, stroka, seq_len, original


# In[12]:


# Подготавливаем данные для обучения - примерно 3Гб
X = []
y = []
for j in speaker_folders1:
    wav_files = list_files_for_speaker(j, voxforge_dir)
    for i in wav_files:
        #i=i.encode('utf-8')
        features, stroka, seq_len, original=extract_features_and_targets(i)
        if seq_len !=0 :
            X.append(features)
            y.append(stroka)


# In[13]:


x=X


# In[14]:


import numpy as np


# In[15]:


x=np.array(x)


# In[16]:


y[1][41]


# In[17]:


type(x[1][6][0])


# In[18]:


stroka


# In[ ]:





# In[19]:


def decode1(stroka,inv_mapping):
    ret = []
    for i in range(len(stroka)):
         ret.append("".join( inv_mapping.get(int(stroka[i]),34)))
    for i in range(len(ret)):
        print(str(ret[i])),
    print('')


# In[20]:


#decode1(stroka,inv_mapping)


# In[21]:


def batch(X_train, y_train, batch_size):
    num_features = X_train[0].shape[1]
    n = len(X_train)
    perm = np.random.permutation(n)
    for batch_ind in np.resize(perm, (n // batch_size, batch_size)):
        X_batch, y_batch = [X_train[i] for i in batch_ind], [y_train[i] for i in batch_ind]
        sequence_lengths = list(map(len, X_batch))
        X_batch_padded = np.array(list(zip_longest(*X_batch, fillvalue=np.zeros(num_features)))).transpose([1, 0, 2])
        yield X_batch_padded, sequence_lengths, list_2d_to_sparse(y_batch), y_batch


# # Модель нейронной сети

# In[22]:


graph = tf.Graph()
with graph.as_default():
    input_X = tf.placeholder(tf.float32, shape=[None, None, 13],name="input_X")
    labels = tf.sparse_placeholder(tf.int32)
    seq_lens = tf.placeholder(tf.int32, shape=[None],name="seq_lens")


    model = Sequential()
    model.add(Bidirectional(LSTM(128, return_sequences=True, implementation=2), input_shape=(None, 13)))
    model.add(Bidirectional(LSTM(128, return_sequences=True, implementation=2)))
    model.add(TimeDistributed(Dense(len(inv_mapping) + 2)))
    
    final_seq_lens = seq_lens

    logits = model(input_X)
    logits = tf.transpose(logits, [1, 0, 2])

    ctc_loss = tf.reduce_mean(tf.nn.ctc_loss(labels, logits, final_seq_lens))
    # ctc_greedy_decoder? merge_repeated=True
    decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, final_seq_lens)
    ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), labels))

    train_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(ctc_loss)


# In[ ]:





# In[23]:


num_epochs=181
epoch_save_step=10


# In[24]:


# Обучаемся в начале на коротких данных 
y_train1=[]
X_train1=[]
for i in range(len(y)):
    if len(y[i]) <75:
        y_train1.append(y[i])
        X_train1.append(X[i])


# Вот отсюда не чего не понятно, код написан на пионе втором , и тензор тоже неизвестно какой версии, я впринцепе большую часть всего перепеал под третий, а дальше не могу понять что происходит

# In[25]:


with tf.Session(graph=graph) as session:
    print("----------------------------------------------------")
    saver = tf.train.Saver(tf.global_variables())
    snapshot = "ctc"
    checkpoint = tf.train.latest_checkpoint(checkpoint_dir="checkpoint1")
    last_epoch = 0
    
    if checkpoint:
        
        print("[i] LOADING checkpoint " + checkpoint)
        try:
            saver.restore(session, checkpoint)
            last_epoch = int(checkpoint.split('-')[-1]) + 1
            print("[i] start from epoch %d" % last_epoch)
        except:
            print("[!] incompatible checkpoint, restarting from 0")
    else:
        # Initializate the weights and biases
        tf.global_variables_initializer().run()


    for epoch in range(last_epoch, num_epochs):
        for X_batch, seq_lens_batch, y_batch, y_batch_orig in batch(X_train1, y_train1, 100):
            feed_dict = {
                input_X: X_batch,
                labels: y_batch,

                seq_lens: seq_lens_batch
            }
            train_loss, train_ler, train_decoded, true, _ = session.run([ctc_loss, ler, decoded[0], labels, train_op], feed_dict=feed_dict)
        if epoch % epoch_save_step == 0 and epoch > 0:
                print("[i] SAVING snapshot %s" % snapshot)
#                 del tf.get_collection_ref ( ' LAYER_NAME_UIDS ' )[ 0 ]
                saver.save(session, "checkpoint1/" + snapshot + ".ckpt", epoch)

#         for X_batch, seq_lens_batch, y_batch, y_batch_orig in batch(X_test, y_test, 4):
#             feed_dict = {
#                 input_X: X_batch,
#                 labels: y_batch,
#                 seq_lens: seq_lens_batch
#             }
#             test_loss, test_ler, test_decoded, true = session.run([ctc_loss, ler, decoded[0], labels], feed_dict=feed_dict)
        print(epoch, train_loss, train_ler)#,  test_loss, test_ler)
        ret=decode(train_decoded, inv_mapping)[:10]
        for i in range(len(ret)):
            print(str(ret[i])),
        print(time.ctime())
        decode1(y_batch_orig[0],inv_mapping)


# Обучаемся на средних данных (длина предложения меньше 181 символа)

# In[ ]:



# y_train1=[]
# X_train1=[]
# for i in range(len(y)):
#     if len(y[i]) <181:
#         y_train1.append(y[i])
#         X_train1.append(X[i])


# Если у Вас мощный компьютер, можно обучиться на полных данных

# In[ ]:


# num_epochs=1000
# epoch_save_step=5
# with tf.Session(graph=graph) as session:

#     saver = tf.train.Saver(tf.global_variables())
#     snapshot = "ctc"
#     checkpoint = tf.train.latest_checkpoint(checkpoint_dir="checkpoint1")
#     last_epoch = 0

#     if checkpoint:
#         print("[i] LOADING checkpoint " + checkpoint)
#         try:
#             saver.restore(session, checkpoint)
#             last_epoch = int(checkpoint.split('-')[-1]) + 1
#             print("[i] start from epoch %d" % last_epoch)
#         except:
#             print("[!] incompatible checkpoint, restarting from 0")
#     else:
#         # Initializate the weights and biases
#         tf.global_variables_initializer().run()


#     for epoch in range(last_epoch, num_epochs):
#         for X_batch, seq_lens_batch, y_batch, y_batch_orig in batch(X, y, 4):
#             feed_dict = {
#                 input_X: X_batch,
#                 labels: y_batch,

#                 seq_lens: seq_lens_batch
#             }
#             train_loss, train_ler, train_decoded, true, _ = session.run([ctc_loss, ler, decoded[0], labels, train_op], feed_dict=feed_dict)
#         if epoch % epoch_save_step == 0 and epoch > 0:
#                 print("[i] SAVING snapshot %s" % snapshot)
# #                 del tf.get_collection_ref ( ' LAYER_NAME_UIDS ' )[ 0 ]
#                 saver.save(session, "checkpoint1/" + snapshot + ".ckpt", epoch)

# #         for X_batch, seq_lens_batch, y_batch, y_batch_orig in batch(X_test, y_test, 4):
# #             feed_dict = {
# #                 input_X: X_batch,
# #                 labels: y_batch,
# #                 seq_lens: seq_lens_batch
# #             }
# #             test_loss, test_ler, test_decoded, true = session.run([ctc_loss, ler, decoded[0], labels], feed_dict=feed_dict)
#         print(epoch, train_loss, train_ler)#,  test_loss, test_ler)
#         ret=decode(train_decoded, inv_mapping)[:10]
#         for i in range(len(ret)):
#             print(str(ret[i])),
#         print(time.ctime())
#         decode1(y_batch_orig[0],inv_mapping)


# In[ ]:




