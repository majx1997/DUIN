# /*
# -*- coding: utf-8 -*-
# /*
# *Copyright (c) 2021, Alibaba Group;
# *Licensed under the Apache License, Version 2.0 (the "License");
# *you may not use this file except in compliance with the License.
# *You may obtain a copy of the License at

# *   http://www.apache.org/licenses/LICENSE-2.0

# *Unless required by applicable law or agreed to in writing, software
# *distributed under the License is distributed on an "AS IS" BASIS,
# *WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# *See the License for the specific language governing permissions and
# *limitations under the License.
# 


from models.base_model import Model
from common.mhsa import *
from common.ssloss import *
from common.Dice import *
from common.utils import *

class DUIN(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE,
                 use_negsampling=False, maxlen=30, maxlen_hard=20, is_training=False, use_ssl=True, T=0.1, A=1, gama=0.5):
        super(DUIN, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE,
                            ATTENTION_SIZE,
                            use_negsampling,
                            maxlen,
                            maxlen_hard,
                            is_training,
                            use_ssl
                            )
        
        self.keep_prob =1.0
        
        with tf.name_scope('Attention_layer'):
            
            clicks_trans_block = SelfAttentionPooling(
                num_heads=2,
                key_mask=self.mask,
                query_mask=self.mask,
                length=maxlen,
                linear_key_dim=HIDDEN_SIZE,
                linear_value_dim=HIDDEN_SIZE,
                output_dim=EMBEDDING_DIM * 2,
                hidden_dim=EMBEDDING_DIM * 4,
                num_layer=1,
                keep_prob=self.keep_prob
            )
            
            item_his_eb_mix = tf.concat([self.item_his_eb, self.item_his_time_eb], axis=2)
            item_his_eb_mix = tf.layers.dense(item_his_eb_mix, EMBEDDING_DIM * 2, activation=None, name='item_his_eb_mix')
            clicks_trans_output = clicks_trans_block.build(item_his_eb_mix, reuse=False, scope='clicks_trans')  # (batch_size, maxlen, output_dim)
              
            item_his_hard_mix = tf.concat([self.item_his_hard_eb, self.item_his_time_hard_eb], axis=2)
            item_his_hard_trans = tf.layers.dense(item_his_hard_mix, self.trigger_eb.get_shape().as_list()[-1], activation=None, name='item_his_hard_mix')
            
            trigger_item = tf.expand_dims(self.trigger_eb, 1)
            trigger_same_eb =  item_his_hard_trans
            trigger_related_sequence = tf.concat([trigger_item, trigger_same_eb], axis=1)
            trigger_related_sequence_mask = tf.sequence_mask(tf.ones_like(trigger_related_sequence[:, 0, 0], dtype=tf.int32), 1)
            trigger_related_sequence_mask = tf.tile(trigger_related_sequence_mask, [1, trigger_related_sequence.get_shape().as_list()[1]])
            
            item_his_hard_target_attention, _ = MHSA(queries=trigger_related_sequence,
                                                    keys=trigger_related_sequence,
                                                    num_units=64,
                                                    num_output_units=self.trigger_eb.get_shape().as_list()[-1],
                                                    activation_fn=None,
                                                    scope="intent_attention",
                                                    reuse=tf.AUTO_REUSE,
                                                    query_masks=trigger_related_sequence_mask,
                                                    key_masks=trigger_related_sequence_mask,
                                                    atten_mode='ln',
                                                    linear_projection=True,
                                                    fix_rtp_bug=True,
                                                    num_heads=8)
            
            intent_vec = item_his_hard_target_attention[:,0,:]

            sequence_mask = tf.sequence_mask(tf.ones_like(clicks_trans_output[:, 0, 0], dtype=tf.int32), 1)
            sequence_mask = tf.tile(sequence_mask, [1, maxlen])
            
            mid_target_graph_tag = tf.concat([self.mid_target_i2i_tag, self.mid_target_i2c_tag, self.mid_target_c2c_tag], axis=2) 
            mid_target_graph_gate = RelevantGate(mid_target_graph_tag, clicks_trans_output.get_shape().as_list()[1], tag='target_mid_graph_tag')
            
            mid_target_seq = mid_target_graph_gate * clicks_trans_output
            
            mid_trigger_graph_tag = tf.concat([self.mid_trigger_i2i_tag, self.mid_trigger_i2c_tag, self.mid_trigger_c2c_tag], axis=2) 
            mid_trigger_graph_gate = RelevantGate(mid_trigger_graph_tag, clicks_trans_output.get_shape().as_list()[1], tag='trigger_mid_graph_tag')
            
            mid_trigger_seq = mid_trigger_graph_gate * clicks_trans_output
            
            item_eb_3d = tf.expand_dims(self.item_eb, 1)
            item_his_target_attention_3d, _ = MHTA(queries=item_eb_3d,
                                                keys=mid_target_seq,
                                                values=mid_target_seq,
                                                num_units=64,
                                                num_output_units=64,
                                                activation_fn=None,
                                                scope="item_his_target_attention",
                                                reuse=tf.AUTO_REUSE,
                                                key_masks=sequence_mask,
                                                atten_mode='ln',
                                                linear_projection=True,
                                                fix_rtp_bug=True,
                                                num_heads=8)
            
            item_his_target_attention = tf.reduce_sum(item_his_target_attention_3d, 1)
            # intent_vec_3d = tf.expand_dims(intent_vec, 1)
            item_his_trigger_attention_3d, _ = MHTA(queries=trigger_item,
                                                keys=mid_trigger_seq,
                                                values=mid_trigger_seq,
                                                num_units=64,
                                                num_output_units=64,
                                                activation_fn=None,
                                                scope="item_his_trigger_attention",
                                                reuse=tf.AUTO_REUSE,
                                                key_masks=sequence_mask,
                                                atten_mode='ln',
                                                linear_projection=True,
                                                fix_rtp_bug=True,
                                                num_heads=8)
            
            item_his_trigger_attention = tf.reduce_sum(item_his_trigger_attention_3d, 1)
                
        if self.use_ssl:
            ## intent SSL
            # trigger_same_eb_aug = tf.cond(tf.reshape(tf.random_uniform([1]), []) < 0.5, lambda:self.random_mask(trigger_same_eb), lambda:self.random_substitute(trigger_same_eb))
            trigger_same_eb_aug = self.random_mask(trigger_same_eb, gama) 
            trigger_related_sequence_aug = tf.concat([trigger_item, trigger_same_eb_aug], axis=1)
            # aug forward
            item_his_hard_target_attention_aug, _ = MHSA(queries=trigger_related_sequence_aug,
                                                        keys=trigger_related_sequence_aug,
                                                        num_units=64,
                                                        num_output_units=self.trigger_eb.get_shape().as_list()[-1],
                                                        activation_fn=None,
                                                        scope="intent_attention",
                                                        reuse=tf.AUTO_REUSE,
                                                        query_masks=trigger_related_sequence_mask,
                                                        key_masks=trigger_related_sequence_mask,
                                                        atten_mode='ln',
                                                        linear_projection=True,
                                                        fix_rtp_bug=True,
                                                        num_heads=8)
            intent_vec_aug = item_his_hard_target_attention_aug[:,0,:]
            self.aux_loss = A * ssl_loss(intent_vec, intent_vec_aug, 
                                    temperature= T,
                                    bs = 256,
                                    stag='intent_ssl',
                                    projector = True,
                                    is_training=self.is_training
                                    )
        else:
            self.aux_loss = tf.constant([0])
            
        with tf.name_scope('IUMM_Layer'):
            
            intention_feature = tf.concat([self.uid_batch_embedded, self.item_his_hard_eb_sum, intent_vec, self.seq_hard_len_eb, self.pagenum_eb], 1)
            
            mu = tf.layers.batch_normalization(inputs=intention_feature, name='mu_bn1', training=self.is_training)
            mu = tf.layers.dense(mu, EMBEDDING_DIM*2, activation=tf.nn.relu, name='mu_1')
            
            mu = tf.layers.batch_normalization(inputs=mu, name='mu_bn2', training=self.is_training)
            mu = tf.layers.dense(mu, 1, activation=None, name='mu')
            
            sigma = tf.layers.batch_normalization(inputs=intention_feature, name='sigma_bn1', training=self.is_training)
            sigma = tf.layers.dense(sigma, EMBEDDING_DIM*2, activation=tf.nn.relu, name='sigma_1')
            
            sigma = tf.layers.batch_normalization(inputs=sigma, name='sigma_bn2', training=self.is_training)
            sigma = tf.layers.dense(sigma, 1, activation=None, name='sigma')
            sigma = tf.nn.softplus(sigma)
            
            intention = tf.compat.v1.distributions.Normal(mu, sigma)
            intention = intention.sample([1])
            intention = tf.squeeze(intention, axis=[0])
            
        with tf.name_scope('Interaction_layer'):
            cross_inp = tf.concat([intent_vec, self.item_eb, intent_vec - self.item_eb, tf.multiply(intent_vec, self.item_eb)], 1)
            cross_inp1 = tf.layers.batch_normalization(inputs=cross_inp, name='interaction_layer_bn1', training=is_training)
            cross_dnn1 = tf.layers.dense(cross_inp1, EMBEDDING_DIM * 4, activation=tf.nn.relu, name='interaction_layer_1')
            cross_inp2 = tf.layers.batch_normalization(inputs=cross_dnn1, name='interaction_layer_bn2', training=is_training)
            cross_dnn2 = tf.layers.dense(cross_inp2, EMBEDDING_DIM * 2, activation=tf.nn.relu, name='interaction_layer_2')

        inp = tf.concat([self.uid_batch_embedded, cross_dnn2, self.trigger_eb, intent_vec, tf.multiply(item_his_trigger_attention, intention), tf.multiply(item_his_target_attention, 1 - intention), self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum], -1)
        # Fully connected layer
        self.build_fcn_net(inp, use_dice=True)
     
    def random_mask(self, inputs, gama):
        mask_ratio = gama
        keep_tensor = mask_ratio + tf.random_uniform((int(inputs.get_shape()[1]), 1))
        mask = tf.cast(tf.floor(keep_tensor), dtype=tf.bool)
        mask = tf.tile(tf.expand_dims(mask, 0), [tf.shape(inputs)[0], 1, tf.shape(inputs)[2]])
        outputs = tf.where(mask, inputs, tf.zeros(shape=tf.shape(inputs)))
        return outputs