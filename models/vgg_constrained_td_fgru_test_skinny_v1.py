#!/usr/bin/env python
import tensorflow as tf
from layers.feedforward import conv, pooling
from layers.feedforward import normalization
from layers.recurrent import constrained_h_td_fgru as hgru


def build_model(data_tensor, reuse, training, output_shape):
    """Create the hgru from Learning long-range..."""
    if isinstance(output_shape, list):
        output_shape = output_shape[0]
    pool_kernel_size = [1, 2, 2, 1]
    pool_kernel_strides = [1, 2, 2, 1]
    with tf.variable_scope('cnn', reuse=reuse):
        # Add input
        in_emb = tf.layers.conv2d(
            inputs=data_tensor,
            filters=16,
            kernel_size=3,
            name='1_conv_1',
            strides=[1, 1],
            padding='same',
            activation=tf.nn.relu,
            trainable=training,
            use_bias=True)
        in_emb = normalization.batch(
            bottom=in_emb,
            renorm=False,
            name='vgghgru_bn_1',
            training=training)
        in_emb = tf.layers.conv2d(
            inputs=in_emb,
            filters=16,
            kernel_size=3,
            name='1_conv_2',
            strides=[1, 1],
            padding='same',
            activation=tf.nn.relu,
            trainable=training,
            use_bias=True)
        in_emb = normalization.batch(
            bottom=in_emb,
            renorm=False,
            name='vgghgru_bn_2',
            training=training)
        in_emb = pooling.max_pool(
            bottom=in_emb,
            name='1_pool_1',
            k=pool_kernel_size,
            s=pool_kernel_strides)
        in_emb = tf.layers.conv2d(
            inputs=in_emb,
            filters=16,
            kernel_size=3,
            name='2_conv_1',
            strides=[1, 1],
            padding='same',
            activation=tf.nn.relu,
            trainable=training,
            use_bias=True)
        in_emb = normalization.batch(
            bottom=in_emb,
            renorm=False,
            name='vgghgru_bn_3',
            training=training)
        in_emb = tf.layers.conv2d(
            inputs=in_emb,
            filters=16,
            kernel_size=3,
            name='2_conv_2',
            strides=[1, 1],
            padding='same',
            activation=None,  # tf.nn.relu,
            trainable=training,
            use_bias=True)
        # in_emb = normalization.batch(
        #     bottom=in_emb,
        #     renorm=False,
        #     name='vgghgru_bn_3',
        #     training=training)
        layer_hgru = hgru.hGRU(
            'fgru',
            x_shape=in_emb.get_shape().as_list(),
            timesteps=8,
            h_ext=[[1, 1], [1, 1], [1, 1]],  # hgru1, hgru2, fgru2, fgru1  # Increase h_ext ~ 15
            strides=[1, 1, 1, 1],
            hgru_ids={'h1': 16, 'h2': 128, 'fb1': 16},  # hGRU labels and channels
            hgru_idx={'h1': 0, 'h2': 1, 'fb1': 2},
            padding='SAME',
            aux={
                'readout': 'fb',
                'intermediate_ff': [64, 128],  # Add one more layer w/ either 36 or 72 channels
                'intermediate_ks': [[7, 7], [5, 5]],
                'while_loop': False,
                'skip': True,
                'include_pooling': True
            },
            pool_strides=[2, 2],
            pooling_kernel=[2, 2],
            train=training)
        h2 = layer_hgru.build(in_emb)
        h2 = normalization.batch(
            bottom=h2,
            renorm=False,
            name='hgru_bn',
            training=training)
        activity = conv.readout_layer(
            activity=h2,
            reuse=reuse,
            training=training,
            output_shape=output_shape)
    extra_activities = {
        'activity': h2
    }
    return activity, extra_activities
