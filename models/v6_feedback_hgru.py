#!/usr/bin/env python
import tensorflow as tf
from layers.feedforward import conv
from layers.feedforward import normalization
from layers.feedforward import pooling
from layers.recurrent import feedback_hgru as hgru


def build_model(data_tensor, reuse, training, output_shape):
    """Create the hgru from Learning long-range..."""
    if isinstance(output_shape, list):
        output_shape = output_shape[0]
    with tf.variable_scope('cnn', reuse=reuse):
        with tf.variable_scope('input', reuse=reuse):
            in_emb = tf.layers.conv2d(
                inputs=data_tensor,
                filters=8,
                kernel_size=11,
                name='l0',
                strides=(1, 1),
                padding='same',
                activation=tf.nn.elu,
                trainable=training,
                use_bias=True)
            in_emb = pooling.max_pool(
                bottom=in_emb,
                name='p1',
                k=[1, 2, 2, 1],
                s=[1, 2, 2, 1])
            in_emb = tf.layers.conv2d(
                inputs=in_emb,
                filters=8,
                kernel_size=7,
                name='l1',
                strides=(1, 1),
                padding='same',
                activation=tf.nn.elu,
                trainable=training,
                use_bias=True)

        with tf.variable_scope('fGRU', reuse=reuse):
            layer_hgru = hgru.hGRU(
                'fgru',
                x_shape=in_emb.get_shape().as_list(),
                timesteps=6,
                h_ext=[[7, 7], [5, 5], [5, 5]],
                strides=[1, 1, 1, 1],
                padding='SAME',
                aux={
                    'readout': 'fb',
                    'intermediate_ff': [8],
                    'intermediate_ks': [[3, 3]],
                },
                pool_strides=[4, 4],
                pooling_kernel=[4, 4],
                train=training)
            x = layer_hgru.build(in_emb)
            x = normalization.batch(
                bottom=x,
                name='hgru_bn',
                fused=True,
                training=training)

        with tf.variable_scope('readout_1', reuse=reuse):
            x = conv.conv_layer(
                bottom=x,
                name='pre_readout_conv',
                num_filters=2,
                kernel_size=1,
                trainable=training,
                use_bias=True)
            pool_aux = {'pool_type': 'max'}
            x = pooling.global_pool(
                bottom=x,
                name='pre_readout_pool',
                aux=pool_aux)
            x = normalization.batch(
                bottom=x,
                name='hgru_bn',
                training=training)

        with tf.variable_scope('readout_2', reuse=reuse):
            pre_activity = tf.layers.flatten(
                x,
                name='flat_readout')
            activity = tf.layers.dense(
                inputs=pre_activity,
                units=output_shape)
    extra_activities = {
        'activity': activity,
    }
    return activity, extra_activities
