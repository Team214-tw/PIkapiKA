import os
import sys
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import keras.backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
from PikaEnv.PikaEnv import PikaEnv

try:
    side = int(sys.argv[1])
except IndexError:
    side = 0
log_dir = f"log/{side}/"
tb_dir = f"tb/{side}/"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(tb_dir, exist_ok=True)
nb_steps = 2_000_000


def main():
    env = PikaEnv()
    nb_actions = env.action_space.n

    model = Sequential()
    model.add(Flatten(input_shape=(4,) + env.observation_space.shape))
    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(Dense(nb_actions))
    model.add(Activation("linear"))
    print(model.summary())
    memory = SequentialMemory(limit=1_000_000, window_length=4)
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr="eps",
        value_max=1.0,
        value_min=0.05,
        value_test=0.05,
        nb_steps=nb_steps // 4,
    )
    dqn = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        policy=policy,
        memory=memory,
        enable_dueling_network=True,
        enable_double_dqn=False,
    )
    dqn.compile(Adam(lr=0.00025), metrics=["mae"])
    # dqn.load_weights(log_dir + "load.h5f")
    weights_filename = log_dir + "dqn_weights.h5f"
    checkpoint_weights_filename = log_dir + "dqn_weights_{step}.h5f"
    log_filename = log_dir + "dqn_log.json"
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=10000)]
    callbacks += [FileLogger(log_filename, interval=100)]
    tbCallBack = TensorBoard(
        log_dir=tb_dir,
        histogram_freq=0,
        write_graph=True,
        write_grads=True,
        write_images=True,
        embeddings_freq=0,
        embeddings_layer_names=None,
        embeddings_metadata=None,
    )
    callbacks += [tbCallBack]
    dqn.fit(
        env,
        callbacks=callbacks,
        nb_steps=nb_steps,
        log_interval=10,
        visualize=True,
        verbose=2,
    )

    # After training is done, we save the final weights one more time.
    dqn.save_weights(weights_filename, overwrite=True)

    # dqn.load_weights(log_dir + "dqn_weights_352000.h5f")
    # dqn.test(env, nb_episodes=10, visualize=True)


if __name__ == "__main__":
    main()
