import numpy as np
import tensorflow as tf
import config


class Model:
    def __init__(self, vocab_size, num_nodes):
        # '''hyperparameter'''
        with tf.name_scope('read_inputs') as scope:
            self.Text_a = tf.placeholder(tf.int32, [config.batch_size, config.MAX_LEN], name='Ta')
            self.Text_b = tf.placeholder(tf.int32, [config.batch_size, config.MAX_LEN], name='Tb')
            self.Text_neg = tf.placeholder(tf.int32, [config.batch_size, config.MAX_LEN], name='Tneg')
            self.Node_a = tf.placeholder(tf.int32, [config.batch_size], name='n1')
            self.Node_b = tf.placeholder(tf.int32, [config.batch_size], name='n2')
            self.Node_neg = tf.placeholder(tf.int32, [config.batch_size], name='n3')

        with tf.name_scope('initialize_embedding') as scope:
            self.text_embed = tf.Variable(tf.truncated_normal([vocab_size, config.embed_size // 2], stddev=0.3))
            self.node_embed = tf.Variable(tf.truncated_normal([num_nodes, config.embed_size // 2], stddev=0.3))
            self.node_embed = tf.clip_by_norm(self.node_embed, clip_norm=1, axes=1)

        with tf.name_scope('lookup_embeddings') as scope:
            self.TA = tf.nn.embedding_lookup(self.text_embed, self.Text_a)
            self.T_A = tf.expand_dims(self.TA, -1)

            self.TB = tf.nn.embedding_lookup(self.text_embed, self.Text_b)
            self.T_B = tf.expand_dims(self.TB, -1)

            self.TNEG = tf.nn.embedding_lookup(self.text_embed, self.Text_neg)
            self.T_NEG = tf.expand_dims(self.TNEG, -1)

            self.N_A = tf.nn.embedding_lookup(self.node_embed, self.Node_a)
            self.N_B = tf.nn.embedding_lookup(self.node_embed, self.Node_b)
            self.N_NEG = tf.nn.embedding_lookup(self.node_embed, self.Node_neg)
        self.convA, self.convB, self.convNeg = self.conv()
        self.loss = self.compute_loss()

    def conv(self):
        W2 = tf.Variable(tf.truncated_normal([2, config.embed_size // 2, 1, 100], stddev=0.3))
        rand_matrix = tf.Variable(tf.truncated_normal([100, 100], stddev=0.3))

        convA = tf.nn.conv2d(self.T_A, W2, strides=[1, 1, 1, 1], padding='VALID')
        convB = tf.nn.conv2d(self.T_B, W2, strides=[1, 1, 1, 1], padding='VALID')
        convNEG = tf.nn.conv2d(self.T_NEG, W2, strides=[1, 1, 1, 1], padding='VALID')

        hA = tf.tanh(tf.squeeze(convA))
        hB = tf.tanh(tf.squeeze(convB))
        hNEG = tf.tanh(tf.squeeze(convNEG))

        tmphA = tf.reshape(hA, [config.batch_size * (config.MAX_LEN - 1), config.embed_size // 2])
        ha_mul_rand = tf.reshape(tf.matmul(tmphA, rand_matrix),
                                 [config.batch_size, config.MAX_LEN - 1, config.embed_size // 2])
        r1 = tf.matmul(ha_mul_rand, hB, adjoint_b=True)
        r3 = tf.matmul(ha_mul_rand, hNEG, adjoint_b=True)
        att1 = tf.expand_dims(tf.stack(r1), -1)
        att3 = tf.expand_dims(tf.stack(r3), -1)

        att1 = tf.tanh(att1)
        att3 = tf.tanh(att3)

        pooled_A = tf.reduce_mean(att1, 2)
        pooled_B = tf.reduce_mean(att1, 1)
        pooled_NEG = tf.reduce_mean(att3, 1)

        a_flat = tf.squeeze(pooled_A)
        b_flat = tf.squeeze(pooled_B)
        neg_flat = tf.squeeze(pooled_NEG)

        w_A = tf.nn.softmax(a_flat)
        w_B = tf.nn.softmax(b_flat)
        w_NEG = tf.nn.softmax(neg_flat)

        rep_A = tf.expand_dims(w_A, -1)
        rep_B = tf.expand_dims(w_B, -1)
        rep_NEG = tf.expand_dims(w_NEG, -1)

        hA = tf.transpose(hA, perm=[0, 2, 1])
        hB = tf.transpose(hB, perm=[0, 2, 1])
        hNEG = tf.transpose(hNEG, perm=[0, 2, 1])

        rep1 = tf.matmul(hA, rep_A)
        rep2 = tf.matmul(hB, rep_B)
        rep3 = tf.matmul(hNEG, rep_NEG)

        attA = tf.squeeze(rep1)
        attB = tf.squeeze(rep2)
        attNEG = tf.squeeze(rep3)

        return attA, attB, attNEG

    def compute_loss(self):
        p1 = tf.reduce_sum(tf.multiply(self.convA, self.convB), 1)
        p1 = tf.log(tf.sigmoid(p1) + 0.001)

        p2 = tf.reduce_sum(tf.multiply(self.convA, self.convNeg), 1)
        p2 = tf.log(tf.sigmoid(-p2) + 0.001)

        p3 = tf.reduce_sum(tf.multiply(self.N_A, self.N_B), 1)
        p3 = tf.log(tf.sigmoid(p3) + 0.001)

        p4 = tf.reduce_sum(tf.multiply(self.N_A, self.N_NEG), 1)
        p4 = tf.log(tf.sigmoid(-p4) + 0.001)

        p5 = tf.reduce_sum(tf.multiply(self.convB, self.N_A), 1)
        p5 = tf.log(tf.sigmoid(p5) + 0.001)

        p6 = tf.reduce_sum(tf.multiply(self.convNeg, self.N_A), 1)
        p6 = tf.log(tf.sigmoid(-p6) + 0.001)

        p7 = tf.reduce_sum(tf.multiply(self.N_B, self.convA), 1)
        p7 = tf.log(tf.sigmoid(p7) + 0.001)

        p8 = tf.reduce_sum(tf.multiply(self.N_B, self.convNeg), 1)
        p8 = tf.log(tf.sigmoid(-p8) + 0.001)

        rho1 = 0.7
        rho2 = 1.0
        rho3 = 0.1
        temp_loss = rho1 * (p1 + p2) + rho2 * (p3 + p4) + rho3 * (p5 + p6) + rho3 * (p7 + p8)
        loss = -tf.reduce_sum(temp_loss)
        return loss
