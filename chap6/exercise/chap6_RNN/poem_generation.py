#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import tensorflow as tf
import collections
from tensorflow import keras
from tensorflow.keras import layers, optimizers

# ============================================
# 1. 数据处理
# ============================================

start_token = 'bos'
end_token = 'eos'

def process_dataset(fileName):
    """处理诗歌数据集"""
    examples = []
    with open(fileName, 'r', encoding='utf-8') as fd:
        for line in fd:
            outs = line.strip().split(':')
            content = ''.join(outs[1:])
            ins = [start_token] + list(content) + [end_token] 
            if len(ins) > 200:
                continue
            examples.append(ins)
            
    counter = collections.Counter()
    for e in examples:
        for w in e:
            counter[w] += 1
    
    sorted_counter = sorted(counter.items(), key=lambda x: -x[1])
    words, _ = zip(*sorted_counter)
    words = ('PAD', 'UNK') + words[:len(words)]
    word2id = dict(zip(words, range(len(words))))
    id2word = {word2id[k]: k for k in word2id}
    
    indexed_examples = [[word2id[w] for w in poem] for poem in examples]
    seqlen = [len(e) for e in indexed_examples]
    
    instances = list(zip(indexed_examples, seqlen))
    
    return instances, word2id, id2word

def poem_dataset():
    """创建数据集"""
    instances, word2id, id2word = process_dataset('./poems.txt')
    ds = tf.data.Dataset.from_generator(lambda: [ins for ins in instances], 
                                            (tf.int64, tf.int64), 
                                            (tf.TensorShape([None]), tf.TensorShape([])))
    ds = ds.shuffle(buffer_size=10240)
    ds = ds.padded_batch(100, padded_shapes=(tf.TensorShape([None]), tf.TensorShape([])))
    ds = ds.map(lambda x, seqlen: (x[:, :-1], x[:, 1:], seqlen-1))
    return ds, word2id, id2word

# ============================================
# 2. 模型定义
# ============================================

class myRNNModel(keras.Model):
    def __init__(self, w2id):
        super(myRNNModel, self).__init__()
        self.v_sz = len(w2id)
        self.embed_layer = tf.keras.layers.Embedding(self.v_sz, 128, 
                                                    batch_input_shape=[None, None])
        
        # 增加到 256 隐元，并堆叠两层 RNN
        self.rnncell_1 = tf.keras.layers.SimpleRNNCell(256)
        self.rnn_layer_1 = tf.keras.layers.RNN(self.rnncell_1, return_sequences=True)
        
        self.rnncell_2 = tf.keras.layers.SimpleRNNCell(256)
        self.rnn_layer_2 = tf.keras.layers.RNN(self.rnncell_2, return_sequences=True)
        
        self.dense = tf.keras.layers.Dense(self.v_sz)
        
    def call(self, inp_ids):
        """前向传播"""
        inp_emb = self.embed_layer(inp_ids)
        x = self.rnn_layer_1(inp_emb)
        x = self.rnn_layer_2(x)
        logits = self.dense(x)
        return logits
    
    def get_next_token(self, x, state_1, state_2):
        """生成下一个字符"""
        inp_emb = self.embed_layer(x)
        h_1, state_1 = self.rnncell_1.call(inp_emb, state_1)
        h_2, state_2 = self.rnncell_2.call(h_1, state_2)
        logits = self.dense(h_2)
        out = tf.argmax(logits, axis=-1)
        return out, state_1, state_2

# ============================================
# 3. 辅助函数
# ============================================

def mkMask(input_tensor, maxLen):
    """创建掩码"""
    shape_of_input = tf.shape(input_tensor)
    shape_of_output = tf.concat(axis=0, values=[shape_of_input, [maxLen]])

    oneDtensor = tf.reshape(input_tensor, shape=(-1,))
    flat_mask = tf.sequence_mask(oneDtensor, maxlen=maxLen)
    return tf.reshape(flat_mask, shape_of_output)

def reduce_avg(reduce_target, lengths, dim):
    """计算加权平均损失"""
    shape_of_lengths = lengths.get_shape()
    shape_of_target = reduce_target.get_shape()
    if len(shape_of_lengths) != dim:
        raise ValueError(('Second input tensor should be rank %d, ' +
                         'while it got rank %d') % (dim, len(shape_of_lengths)))
    if len(shape_of_target) < dim+1:
        raise ValueError(('First input tensor should be at least rank %d, ' +
                         'while it got rank %d') % (dim+1, len(shape_of_target)))

    rank_diff = len(shape_of_target) - len(shape_of_lengths) - 1
    mxlen = tf.shape(reduce_target)[dim]
    mask = mkMask(lengths, mxlen)
    if rank_diff != 0:
        len_shape = tf.concat(axis=0, values=[tf.shape(lengths), [1]*rank_diff])
        mask_shape = tf.concat(axis=0, values=[tf.shape(mask), [1]*rank_diff])
    else:
        len_shape = tf.shape(lengths)
        mask_shape = tf.shape(mask)
    lengths_reshape = tf.reshape(lengths, shape=len_shape)
    mask = tf.reshape(mask, shape=mask_shape)

    mask_target = reduce_target * tf.cast(mask, dtype=reduce_target.dtype)

    red_sum = tf.reduce_sum(mask_target, axis=[dim], keepdims=False)
    red_avg = red_sum / (tf.cast(lengths_reshape, dtype=tf.float32) + 1e-30)
    return red_avg

# ============================================
# 4. 损失和训练函数
# ============================================

def compute_loss(logits, labels, seqlen):
    """计算损失"""
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels)
    losses = reduce_avg(losses, seqlen, dim=1)
    return tf.reduce_mean(losses)

@tf.function(experimental_relax_shapes=True)
def train_one_step(model, optimizer, x, y, seqlen):
    """执行一步训练"""
    with tf.GradientTape() as tape:
        logits = model(x)
        loss = compute_loss(logits, y, seqlen)
    
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

def train(epoch, model, optimizer, ds):
    """训练一个 epoch"""
    loss = 0.0
    for step, (x, y, seqlen) in enumerate(ds):
        loss = train_one_step(model, optimizer, x, y, seqlen)
        if step % 10 == 0:
            print(f'epoch {epoch} : step {step} : loss {loss.numpy():.4f}')
    return loss

# ============================================
# 5. 生成函数
# ============================================

def gen_sentence():
    """生成随机诗歌"""
    state_1 = [tf.random.normal(shape=(1, 256), stddev=0.5), 
               tf.random.normal(shape=(1, 256), stddev=0.5)]
    state_2 = [tf.random.normal(shape=(1, 256), stddev=0.5), 
               tf.random.normal(shape=(1, 256), stddev=0.5)]
    cur_token = tf.constant([word2id['bos']], dtype=tf.int32)
    collect = []
    for _ in range(50):
        cur_token, state_1, state_2 = model.get_next_token(cur_token, state_1, state_2)
        collect.append(cur_token.numpy()[0])
    return ''.join([id2word[t] for t in collect])

def gen_sentence_with_start(start_word):
    """以指定开头词汇生成诗歌"""
    state_1 = [tf.random.normal(shape=(1, 256), stddev=0.5), 
               tf.random.normal(shape=(1, 256), stddev=0.5)]
    state_2 = [tf.random.normal(shape=(1, 256), stddev=0.5), 
               tf.random.normal(shape=(1, 256), stddev=0.5)]
    
    start_id = word2id.get(start_word, word2id['UNK'])
    cur_token = tf.constant([start_id], dtype=tf.int32)
    
    collect = [start_word]
    
    for _ in range(49):
        cur_token, state_1, state_2 = model.get_next_token(cur_token, state_1, state_2)
        char_id = cur_token.numpy()[0]
        char = id2word.get(char_id, 'UNK')
        collect.append(char)
        
        if char == 'eos':
            break
    
    return ''.join(collect)

# ============================================
# 6. 主程序
# ============================================

if __name__ == '__main__':
    print("=" * 60)
    print("RNN 诗歌生成程序")
    print("=" * 60)
    
    # 禁用冗余警告
    tf.get_logger().setLevel('ERROR')
    
    print("\n[1/4] 加载数据...")
    train_ds, word2id, id2word = poem_dataset()
    print(f"      词汇表大小: {len(word2id)}")
    
    print("\n[2/4] 创建模型...")
    model = myRNNModel(word2id)
    optimizer = optimizers.Adam(0.001)  # 提高学习率
    print("      模型已创建（双层 RNN，隐层 256，嵌入 128）")
    
    print("\n[3/4] 开始训练（10个epoch）...")
    for epoch in range(10):
        loss = train(epoch, model, optimizer, train_ds)
        print(f"      Epoch {epoch} 完成，最终损失: {loss.numpy():.4f}")
    
    print("\n[4/4] 生成诗歌...")
    print("=" * 60)
    print("随机生成（无指定开头）：")
    print("=" * 60)
    random_poem = gen_sentence()
    print(random_poem)
    
    print("\n" + "=" * 60)
    print("指定开头词汇生成结果：")
    print("=" * 60)
    
    start_words = ['日', '红', '山', '夜', '湖', '海', '月']
    
    for word in start_words:
        result = gen_sentence_with_start(word)
        print(f"\n开头词：{word}")
        print(f"生成诗歌：{result}")
        print("-" * 60)

    print("\n✓ 程序执行完成！")
