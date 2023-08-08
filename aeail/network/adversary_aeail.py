from aeail.baselines.common.mpi_running_mean_std import RunningMeanStd
import tensorflow as tf
from aeail.baselines.common import tf_util as U
from aeail.common.tf_util import *
import numpy as np
import ipdb

class TransitionClassifier(object):
  def __init__(self, env, hidden_size, entcoeff=0.001, lr_rate=1e-3, scope="adversary"):
    self.scope = scope
    self.observation_shape = env.observation_space.shape
    self.actions_shape = env.action_space.shape
    self.input_shape = tuple([o+a for o,a in zip(self.observation_shape, self.actions_shape)])
    self.num_actions = env.action_space.shape[0]
    self.hidden_size = hidden_size
    self.build_ph()
    # Build grpah
    generator_logits, generator_latent_code = self.build_graph(self.generator_obs_ph, self.generator_acs_ph, reuse=False)
    expert_logits, expert_latent_code = self.build_graph(self.expert_obs_ph, self.expert_acs_ph, reuse=True)
    # Build accuracy

    discriminator_loss = tf.reduce_mean(-1/(1+expert_logits) + 1/(1+generator_logits))

    self.losses = [discriminator_loss]
    self.loss_name = ["discriminator_loss"]
    self.total_loss = discriminator_loss

    self.reward_op = 1/(1+generator_logits)

    self.code = generator_latent_code

    var_list = self.get_trainable_variables()
    self.lossandgrad = U.function([self.generator_obs_ph, self.generator_acs_ph, self.expert_obs_ph, self.expert_acs_ph], 
                         self.losses + [U.flatgrad(self.total_loss, var_list)])

  def build_ph(self):
    self.generator_obs_ph = tf.placeholder(tf.float32, (None, ) + self.observation_shape, name="observations_ph")
    self.generator_acs_ph = tf.placeholder(tf.float32, (None, ) + self.actions_shape, name="actions_ph")
    self.expert_obs_ph = tf.placeholder(tf.float32, (None, ) + self.observation_shape, name="expert_observations_ph")
    self.expert_acs_ph = tf.placeholder(tf.float32, (None, ) + self.actions_shape, name="expert_actions_ph")

  def build_graph(self, obs_ph, acs_ph, reuse=False):
    with tf.variable_scope(self.scope):
      if reuse:
        tf.get_variable_scope().reuse_variables()

      with tf.variable_scope("obfilter"):
          self.obs_rms = RunningMeanStd(shape=self.observation_shape)
      obs = (obs_ph - self.obs_rms.mean) / self.obs_rms.std
      _input = tf.concat([obs, acs_ph], axis=1) # concatenate the two input -> form a transition
      p_h1 = tf.contrib.layers.fully_connected(_input, 100, activation_fn=tf.nn.tanh)
      p_h2 = tf.contrib.layers.fully_connected(p_h1, 100, activation_fn=tf.nn.tanh)
      p_h3 = tf.contrib.layers.fully_connected(p_h2, self.observation_shape[0]+self.actions_shape[0], activation_fn=tf.identity)
      mse = tf.reduce_sum((p_h3 - tf.concat([obs_ph, acs_ph], axis=1))**2, 1)
        
    return mse, p_h1

  def get_trainable_variables(self):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

  def get_reward(self, obs, acs):
    sess = U.get_session()
    if len(obs.shape) == 1:
      obs = np.expand_dims(obs, 0)
    if len(acs.shape) == 1:
      acs = np.expand_dims(acs, 0)
    feed_dict = {self.generator_obs_ph:obs, self.generator_acs_ph:acs}
    reward = sess.run(self.reward_op, feed_dict)
    return reward

  def get_latent_code(self, obs, acs):
    sess = U.get_session()
    if len(obs.shape) == 1:
      obs = np.expand_dims(obs, 0)
    if len(acs.shape) == 1:
      acs = np.expand_dims(acs, 0)
    feed_dict = {self.generator_obs_ph: obs, self.generator_acs_ph: acs}
    code = sess.run(self.code, feed_dict)
    return code


