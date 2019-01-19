#!/usr/bin/env python
import tensorflow as tf
from layers.feedforward import conv
from layers.feedforward import normalization
from layers.recurrent import h_td_fgru as hgru


def build_model(data_tensor, reuse, training, output_shape):
    """Create the hgru from Learning long-range..."""
    if isinstance(output_shape, list):
        output_shape = output_shape[0]
    with tf.variable_scope('cnn', reuse=reuse):
        # Add input
        in_emb = conv.skinny_input_layer(
            X=data_tensor,
            reuse=reuse,
            training=training,
            features=14,
            conv_kernel_size=7,
            pool=False,
            name='l0')
        layer_hgru = hgru.hGRU(
            'fgru',
            x_shape=in_emb.get_shape().as_list(),
            timesteps=8,
            h_ext=[[9, 9], [1, 1], [1, 1]],  # hgru1, hgru2, fgru2, fgru1  # Increase h_ext ~ 15
            strides=[1, 1, 1, 1],
            hgru_ids={'h1': 14, 'h2': 48, 'fb1': 14},  # hGRU labels and channels
            hgru_idx={'h1': 0, 'h2': 1, 'fb1': 2},
            padding='SAME',
            aux={
                'readout': 'fb',
                'intermediate_ff': [24, 36, 48],  # Add one more layer w/ either 36 or 72 channels
                'intermediate_ks': [[5, 5], [5, 5], [3, 3]],
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
