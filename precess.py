import sys, pickle, os, random
import numpy as np
import tensorflow as tf
from tensorflow.contrib.crf import viterbi_decode
from tensorflow.contrib.crf import crf_log_likelihood, crf_sequence_score
from tensorflow.contrib.rnn import LSTMCell
from config import run_time

class BiLSTMCRF():

    def __init__(self,  use_pretrained_embeddings=True, embedding_dim=100, update_embedding=True,
                  hidden_dim = [150],
                  if_multi_layer=0
                 ):
        """
        mode: 制定训练模式或者预测模式
        embedding行：词向量相关配置
        """
        self.model = 'train'
        print("开始初始化")
        self.word_id_map = pickle.load(open())
        print("词汇表大小", len(self.word_id_map))
        self.word_id_map = {v: k for k ,v in self.word_id_map.items()}
        self.init_embeddings(use_pretrained_embeddings, embedding_dim)
        self.bulid_graph(hidden_dim=hidden_dim, if_multi_layer=if_multi_layer)


    def init_embedding(self, use_pretrained_embeddings, embedding_dim):
        print("初始化词向量。")
        if self.mode == 'train':
             if use_pretrained_embeddings ==True:
                 print("读取预训练的词向量")
                 self.embeddings = pickle.load(open())

             else:
                  print("随机初始化一份词向量")
                  self.embeddings = np.float32(np.random.uniform(-0.25, 0.25,(len(self.word_id_map),embedding_dim)))
        else:
            print("加载自己的词向量")
            self.embeddings = pickle(open())
        print("词向量shape",self.embeddings)

        with tf.variable_scope("words"):
            print(self.embeddings)
            self._word_embedddings = tf.Variable(self.embeddings, dtype=tf.float32, trainable=True,
                                           name="_word_embeddings")

    def bulid_graph(self, clip_grad=5.0, hidden_dim=[150], if_multi_layer= 0):
        # 创建输入变量
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="id_of_words")
        self.real_labels = tf.placeholder(tf.int32, shape=[None, None], name="real_labels")
        self.seq_length = tf.placeholder(tf.int32, shape=[None, None], name="length_of_sentence")
        self.lr = tf.placeholder(tf.float16, shape=[], name="learning_rate")  # 便于细致地训练模型
        self.batch_size = tf.placeholder(tf.int32, shape=[], name="batch_size")  # 便于细致地训练模型
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")

        word_embeddings = tf.nn.embedding_lookup(params=self._word_embeddings, ids=self.word_ids,
                                                 name="word_embeddings")  # 从词向量矩阵中，为词语找到对应的词向量，形成序列
        # 词向量参数较多；由于语料等原因，噪声比较多。需要dropout,避免过拟合
        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout)


        #lstm层
        if if_multi_layer==1:
            self.multi_BiLSTM_layer(hidden_dim_list=hidden_dim)
        else:
            self.BiLSTM_layer(hidden_dim[0])
        self.logits = self.attention(self.logits)
        self.CRF_layer()


        #优化器
        optim = tf.train.AdamOptimizer(learning_rate=self.lr)
        grads_and_vars = optim.compute_gradients(self.loss)
        grads_and_vars_clip = [[tf.clip_by_value(g, -clip_grad, clip_grad), v] for g, v in grads_and_vars]
        self.train_op = optim.apply_gradients(grads_and_vars_clip)


        #初始化图中的变量
        config = tf.ConfigProto()
        #config.gpu_options.allow_growth = True
        #config.gpu_options.per_process_gpu_memory_fraction = 0.7  # need ~700MB GPU memory
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())


        #CRF层
    def CRF_layer(self, hidden_dim):
        log_likelihood, self.transition_params = crf_log_likelihood(input=self.logits,tag_indices=self.real_labels,
                                                                    sequence_lengths=self.seq_length)
        self.loss = tf.reduce_mean(log_likelihood)

    def BiLSTM_layer(self, hidden_dim):
        with tf.variable_scope("bilstm_" + str(0)):
            cell_fw, cell_bw = LSTMCell(hidden_dim), LSTMCell(hidden_dim)
            (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw, cell_bw=cell_bw, inputs=self.word_embeddings, sequence_length=self.seq_length,
                dtype=tf.float32)
            output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
            output = tf.nn.dropout(output, self.dropout)

        with tf.variable_scope("proj_" + str(0)):
            self.W = tf.get_variable(name="W" + str(0),
                  shape=[2 * hidden_dim, run_time.TAG_NUM],
                                    initializer=tf.contrib.layers.xavier_initilizer(), dtype=tf.float32 )
            self.b = tf.get_variable(name="b" + str(0),
                   shape=[run_time.TAG_NUM],initializer=tf.zeros_initilizer(), dtype=tf.float32 )
        s= tf.shape(output)
        output = tf.shape(output,[-1, 2*hidden_dim])
        pred = tf.matmul(output,self.W) + self.b
        self.logits = tf.reshape(pred, [-1, s[1], run_time.TAG_NUM])

    def muti_BiLSTM_layer(self, hidden_dim_list = [50,50,50]):
        print("lstm结构是",hidden_dim_list)
        inversed_inputs = tf.reverse_sequence(self.word_embeddings, self.seq_length, batch_dim=0, seq_axis=1)
        def attn_cell(n_hidden):
            lstm_cell = LSTMCell(n_hidden, forget_bias=0.8)
            return tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob = self.dropout)

        cell_fw_list, cell_bw_list = [],[]
        for hidden_dim in hidden_dim_list:
            cell_fw_list.append(attn_cell(hidden_dim))
            cell_bw_list.append(attn_cell(hidden_dim))
        print("lstm的层数是", len(cell_bw_list))
        mlstm_cell_fw = tf.nn.rnn_cell.MultiRNNCell(cell_fw_list, state_is_tuple= True)
        mlstm_cell_bw = tf.nn.rnn_cell.MultiRNNCell(cell_bw_list, state_is_tuple= True)
        initial_state_fw = mlstm_cell_fw.zero_state(self.batch_size, dtype=tf.float32)
        initial_state_bw = mlstm_cell_bw.zero_state(self.batch_size, dtype=tf.float32)

        with tf.variable_scope("bilstm_fw"): #正向
            output_fw_0, _ = tf.nn.dynamic_rnn(cell=mlstm_cell_fw, inputs = self.word_embeddings,
                        initial_state = initial_state_fw, sequence_length=self.seq_length,dtype = tf.float32)
        with tf.variable_scope("bilstm_bw"):
            output_bw_0, _ = tf.nn.dynamic_rnn(cell=mlstm_cell_bw,inputs = inversed_inputs,
                        initial_state = initial_state_bw, sequence_length=self.seq_length,dtype = tf.float32)
        output = tf.concat([output_fw_0, output_bw_0],axis= -1)
        layer_no = 0
        s = tf.shape(output)
        with tf.variable_scope("proj_" + str(layer_no)):
            self.W = tf.get_variable(name='W_'+ str(layer_no),shape=[2 * hidden_dim, run_time.TAG_NUM],
                                  initializer= tf.contrib.layers.xavier_initilizer(), dtype=tf.float32
                                     )
            self.b = tf.get_variable(name="b_" + str(layer_no), shape=[run_time.TAG_NUM],
                                     initializer=tf.contrib.layers.xavier_initilizer(), dtype=tf.float32)
            output = tf.reshape(output,[-1, 2*hidden_dim])
            pred = tf.matmul(output,self.W) + self.b
            self.logits = tf.reshape(pred, [-1, s[1], run_time.TAG_NUM])

    def attention(self, inputs, attention_size=100, time_major=False):
        if isinstance(inputs, tuple):
            inputs = tf.concat(inputs, 2)
        if time_major:
            inputs = tf.transpose(inputs, [1,0,2])
        hidden_size = inputs.shape[2].value
        w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev= 0.1))
        b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        v = tf.tanh(tf.tensordot(inputs,w_omega, axes=1) + b_omega)
        vu = tf.tensordot(v, u_omega,axes=1,name='vu')
        alphas = tf.nn.softmax(vu, name='alphas')

        output = tf.multiply(inputs, tf.expand_dims(alphas, -1))
        output = tf.add(output,inputs)
        return output