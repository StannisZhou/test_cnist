import tensorflow as tf


def apply_normalization(
    activity,
    normalization_type,
    data_format,
    training,
    reuse,
    name,
    trainable,
    scale=True,
    center=True,
    scope=None,
):
    """Route to appropriate normalization."""

    def norm_fun(
        activity,
        normalization_type,
        data_format,
        training,
        trainable,
        reuse,
        scope,
        scale,
        center,
        name,
    ):
        """Apply selected normalization."""
        if reuse == tf.AUTO_REUSE:
            reuse = True
        if normalization_type is 'batch_norm':
            return batch_contrib(
                reuse=reuse,
                bottom=activity,
                renorm=False,
                name=name,
                dtype=activity.dtype,
                data_format=data_format,
                trainable=trainable,
                scale=scale,
                center=center,
                scope=scope,
                training=training,
            )
        elif normalization_type is 'instance_norm':
            return instance(
                reuse=reuse,
                bottom=activity,
                data_format=data_format,
                scale=scale,
                center=center,
                scope=scope,
                training=training,
            )
        elif normalization_type is 'no_param_batch_norm':
            return batch_contrib(
                reuse=reuse,
                bottom=activity,
                renorm=False,
                name=name,
                dtype=activity.dtype,
                data_format=data_format,
                trainable=trainable,
                scale=False,
                center=False,
                scope=scope,
                training=training,
            )
        elif normalization_type is 'no_param_instance_norm':
            return instance(
                reuse=reuse,
                bottom=activity,
                data_format=data_format,
                scale=False,
                center=False,
                scope=scope,
                training=training,
            )
        elif normalization_type is 'ada_batch_norm':
            return batch_contrib(
                reuse=reuse,
                bottom=activity,
                renorm=False,
                name=name,
                dtype=activity.dtype,
                data_format=data_format,
                trainable=trainable,
                scale=scale,
                center=center,
                scope=scope,
                training=training,
            )
        elif normalization_type is 'batch_norm_original':
            return batch(
                reuse=reuse,
                bottom=activity,
                renorm=False,
                momentum=0.95,
                name=name,
                dtype=activity.dtype,
                data_format=data_format,
                trainable=trainable,
                scale=scale,
                center=center,
                training=training,
            )
        elif normalization_type is 'no_param_batch_norm_original':
            return batch(
                reuse=reuse,
                bottom=activity,
                renorm=False,
                momentum=0.95,
                name=name,
                dtype=activity.dtype,
                data_format=data_format,
                trainable=trainable,
                scale=False,
                center=False,
                training=training,
            )
        else:
            raise NotImplementedError(normalization_type)

    if scope is None:
        with tf.variable_scope(name, reuse=reuse) as scope:
            activity = norm_fun(
                activity=activity,
                normalization_type=normalization_type,
                data_format=data_format,
                training=training,
                trainable=trainable,
                reuse=reuse,
                scale=scale,
                center=center,
                name=name,
                scope=scope,
            )
    else:
        activity = norm_fun(
            activity=activity,
            normalization_type=normalization_type,
            data_format=data_format,
            training=training,
            trainable=trainable,
            reuse=reuse,
            scale=scale,
            center=center,
            name=name,
            scope=scope,
        )
    return activity


def batch(
    bottom,
    name,
    scale=True,
    center=True,
    fused=False,
    renorm=False,
    data_format='NHWC',
    dtype=tf.float32,
    reuse=False,
    training=True,
):
    """Wrapper for layers batchnorm."""
    if data_format == 'NHWC' or data_format == 'channels_last':
        axis = -1
    elif data_format == 'NCHW' or data_format == 'channels_first':
        axis = 1
    else:
        raise NotImplementedError(data_format)
    return tf.layers.batch_normalization(
        inputs=bottom,
        name=name,
        scale=scale,
        center=center,
        beta_initializer=tf.zeros_initializer(dtype=dtype),
        gamma_initializer=tf.ones_initializer(dtype=dtype),
        moving_mean_initializer=tf.zeros_initializer(dtype=dtype),
        moving_variance_initializer=tf.ones_initializer(dtype=dtype),
        fused=fused,
        renorm=renorm,
        reuse=reuse,
        axis=axis,
        training=training,
    )


def instance(
    bottom,
    scale=True,
    center=True,
    data_format='NHWC',
    dtype=tf.float32,
    reuse=False,
    scope=None,
    training=True,
):
    """Wrapper for layers batchnorm."""
    if data_format is not 'NHWC' or data_format is not 'channels_last':
        pass
    elif data_format is not 'NCHW' or data_format is not 'channels_first':
        pass
    else:
        raise NotImplementedError(data_format)
    # param_initializer = {
    #     'moving_mean': tf.constant_initializer(0., dtype=dtype),
    #     'moving_variance': tf.constant_initializer(1., dtype=dtype),
    #     'gamma': tf.constant_initializer(0.1, dtype=dtype)
    # }
    return tf.contrib.layers.instance_norm(
        inputs=bottom,
        scale=scale,
        center=center,
        # param_initializers=param_initializer,
        reuse=reuse,
        scope=scope,
        data_format=data_format,
        trainable=training,
    )


def batch_contrib(
    bottom,
    name,
    scale=True,
    center=True,
    fused=None,
    renorm=False,
    recurrent_scale=False,
    dtype=tf.float32,
    data_format='NHWC',
    reuse=False,
    scope=None,
    training=True,
    trainable=None,
):
    """Wrapper for contrib layers batchnorm."""
    if trainable is None:
        trainable = training
    param_initializer = {
        'moving_mean': tf.constant_initializer(0.0, dtype=dtype),
        'moving_variance': tf.constant_initializer(1.0, dtype=dtype),
    }
    if recurrent_scale:
        param_initializer['gamma'] = tf.constant_initializer(0.1, dtype=dtype)
    else:
        param_initializer['gamma'] = tf.constant_initializer(1.0, dtype=dtype)
    return tf.contrib.layers.batch_norm(
        inputs=bottom,
        scale=scale,
        center=center,
        param_initializers=param_initializer,
        updates_collections=None,
        data_format=data_format,
        fused=fused,
        renorm=renorm,
        scope=scope,
        trainable=trainable,
        is_training=training,
    )
