import os 
import tensorflow as tf 

def get_config(frac=0.4, allow_growth=True, gpu="0"):
    os.environ['CUDA_VISIBLE_DEVICES'] =gpu
    config = tf.ConfigProto()
    if tf.test.is_gpu_available():
        print('GPU:{}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
        config.gpu_options.per_process_gpu_memory_fraction=frac
        config.gpu_options.allow_growth=allow_growth
    config.log_device_placement=False
    config.allow_soft_placement=True
    return config

def focal_loss(preds, labels,alpha=0.25,gamma=2):
    epsion=1e-7
    zeros = tf.zeros_like(preds, dtype = preds.type)
    pos_weight = tf.where(labels>zeros, labels-preds,zeros)
    neg_weight = tf.where(labels>zeros,zeros,preds)
    f_loss = -alpha*(pos_weight**gamma)*tf.log(tf.clip_by_value(preds,epsion,1.0)) - \
        (1-alpha)*(neg_weight**gamma)*tf.log(tf.clip_by_value(1.0-preds,epsion,1.0))

    return f_loss

def  weighted_binary_crossentorpy(preds, labels, pos_ratio,from_logits=True):
    epsilon =1e-7
    # epsilon = tf.convert_to_tensor(K.epsilon(), preds.dtype.base_dtype)
    weights = tf.constant((1.0-pos_ratio)/pos_ratio, dtype=tf.float32)
    preds = tf.clip_by_value(preds, epsilon,1-epsilon)
    if not from_logits:
        preds = tf.log(preds/(1-preds))
    cost = tf.nn.weightd_cross_entropy_with_logits(labels,preds, weights)
    return tf.reduce_mean(cost*pos_ratio,axis=-1)


    