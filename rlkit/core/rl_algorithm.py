import abc
from collections import OrderedDict
import time
import os

import gtimer as gt
import numpy as np
import queue
import multiprocessing as mp
try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass

from rlkit.core import eval_util
from rlkit.core.process_spawner import ProcessSpawner
from rlkit.data_management.env_replay_buffer import MultiTaskReplayBuffer
from rlkit.data_management.path_builder import PathBuilder
from rlkit.samplers.in_place import InPlacePathSampler


class MetaRLAlgorithm(metaclass=abc.ABCMeta):
    def __init__(
            self,
            env,
            agent,
            train_tasks,
            eval_tasks,
            meta_batch=64,
            num_iterations=100,
            num_train_steps_per_itr=1000,
            num_initial_steps=100,
            num_tasks_sample=100,
            num_steps_prior=100,
            num_steps_posterior=100,
            num_extra_rl_steps_posterior=100,
            eval_interval=1,
            num_evals=1,
            num_steps_per_eval=1000,
            num_task_eval = None,
            batch_size=1024,
            embedding_batch_size=1024,
            embedding_mini_batch_size=1024,
            max_path_length=1000,
            discount=0.99,
            replay_buffer_size=1000000,
            reward_scale=1,
            num_exp_traj_eval=1,
            update_post_train=1,
            eval_deterministic=True,
            render=False,
            save_replay_buffer=False,
            save_algorithm=False,
            save_environment=False,
            render_eval_paths=False,
            dump_eval_paths=False,
            skip_init_data_collection=False,
            parallelize_data_collection=False,
            plotter=None,
            loggers=None,
            algo_params=None,
            env_params=None,
            latent_dim=None,
            net_size=None,
    ):
        """
        :param env: training env
        :param agent: agent that is conditioned on a latent variable z that rl_algorithm is responsible for feeding in
        :param train_tasks: list of tasks used for training
        :param eval_tasks: list of tasks used for eval
        see default experiment config file for descriptions of the rest of the arguments
        """
        self.env = env
        self.agent = agent
        self.exploration_agent = agent # Can potentially use a different policy purely for exploration rather than also solving tasks, currently not being used
        self.train_tasks = train_tasks
        self.eval_tasks = eval_tasks
        self.meta_batch = meta_batch
        self.num_iterations = num_iterations
        self.num_train_steps_per_itr = num_train_steps_per_itr
        self.num_initial_steps = num_initial_steps
        self.num_tasks_sample = num_tasks_sample
        self.num_steps_prior = num_steps_prior
        self.num_steps_posterior = num_steps_posterior
        self.num_extra_rl_steps_posterior = num_extra_rl_steps_posterior
        self.eval_interval = eval_interval
        self.num_evals = num_evals
        self.num_steps_per_eval = num_steps_per_eval
        self.num_task_eval = num_task_eval
        if self.num_task_eval is None:
            self.num_task_eval = len(self.eval_tasks)
        self.batch_size = batch_size
        self.embedding_batch_size = embedding_batch_size
        self.embedding_mini_batch_size = embedding_mini_batch_size
        self.max_path_length = max_path_length
        self.discount = discount
        self.replay_buffer_size = replay_buffer_size
        self.reward_scale = reward_scale
        self.update_post_train = update_post_train
        self.num_exp_traj_eval = num_exp_traj_eval
        self.eval_deterministic = eval_deterministic
        self.render = render
        self.save_replay_buffer = save_replay_buffer
        self.save_algorithm = save_algorithm
        self.save_environment = save_environment

        self.eval_statistics = None
        self.train_statistics = None
        self.render_eval_paths = render_eval_paths
        self.dump_eval_paths = dump_eval_paths
        self.skip_init_data_collection = skip_init_data_collection
        self.plotter = plotter
        self.loggers = loggers

        self.sampler = InPlacePathSampler(
            env=env,
            policy=agent,
            max_path_length=self.max_path_length,
        )

        # separate replay buffers for
        # - training RL update
        # - training encoder update
        self.replay_buffer = MultiTaskReplayBuffer(
                self.replay_buffer_size,
                env,
                self.train_tasks,
            )

        self.enc_replay_buffer = MultiTaskReplayBuffer(
                self.replay_buffer_size,
                env,
                self.train_tasks,
        )

        self._n_env_steps_total = 0
        self._n_train_steps_total = 0
        self._n_rollouts_total = 0
        self._do_train_time = 0
        self._epoch_start_time = None
        self._algo_start_time = None
        self._old_table_keys = None
        self._eval_old_table_keys = None
        self._current_path_builder = PathBuilder()
        self._exploration_paths = []

        self.parallelize_data_collection = parallelize_data_collection
        if self.parallelize_data_collection:
            print("WE ARE PARALLEL")
            self.buffer_queue = mp.Queue()
            self.weight_queue = mp.Queue()
            self.replay_buffer_dict_key = "replay"
            self.enc_replay_buffer_dict_key = "enc"
            self.collect_data_process = None
            self.n_env_steps_shared = mp.Value('i', 0)
            self.mean_return_shared = mp.Value('d', 0)
            self.mean_final_return_shared = mp.Value('d', 0)
            self.status_shared = mp.Value('i', 0)  # 0 is data being collected, 1 is data finished collected

            self.process_spawner = ProcessSpawner(self.buffer_queue,
                                                  self.weight_queue,
                                                  self.mean_return_shared,
                                                  self.mean_final_return_shared,
                                                  self.n_env_steps_shared,
                                                  self.status_shared,
                                                  self.train_tasks,
                                                  self.max_path_length,
                                                  self.num_tasks_sample,
                                                  self.num_steps_prior,
                                                  self.num_steps_posterior,
                                                  self.num_extra_rl_steps_posterior,
                                                  self.update_post_train,
                                                  self.replay_buffer_dict_key,
                                                  self.enc_replay_buffer_dict_key,
                                                  self.embedding_batch_size,
                                                  algo_params, env_params, latent_dim, net_size)


    def make_exploration_policy(self, policy):
         return policy

    def make_eval_policy(self, policy):
        return policy

    def sample_task(self, is_eval=False):
        '''
        sample task randomly
        '''
        if is_eval:
            idx = np.random.randint(len(self.eval_tasks))
        else:
            idx = np.random.randint(len(self.train_tasks))
        return idx

    def train(self):
        '''
        meta-training loop
        '''
        logger = self.loggers[0]
        if self.dump_eval_paths:
            logger.save_extra_data(self.env.get_tasks(), path='tasks')
        self.pretrain()
        gt.reset()
        gt.set_def_unique(False)
        self._current_path_builder = PathBuilder()

        # at each iteration, we first collect data from tasks, perform meta-updates, then try to evaluate
        for it_ in gt.timed_for(
                range(self.num_iterations),
                save_itrs=True,
        ):
            self._start_epoch(it_)
            self.training_mode(True)

            # save model and optimizer parameters
            params = self.get_epoch_snapshot(-1)
            logger.save_itr_params(-1, params)
            # optionally save the current replay buffer
            if self.save_replay_buffer:
                logger.save_data_with_torch(self.replay_buffer, path='replay_buffer')
                logger.save_data_with_torch(self.enc_replay_buffer, path='enc_replay_buffer')

            # initial data collection
            if it_ == 0 and not self.skip_init_data_collection:
                print('collecting initial data buffer for all train tasks')
                for idx in self.train_tasks:
                    print('task: {} / {}'.format(idx, len(self.train_tasks)))
                    self.task_idx = idx
                    self.env.reset_task(idx)
                    self.collect_data(self.num_initial_steps, 1, np.inf)
                # save the initial replay buffer
                print('saving the initial replay buffer')
                logger.save_data_with_torch(self.replay_buffer, path='init_buffer')
                exit()

            # sample data from train tasks
            print('epoch: {}, sampling training data'.format(it_))
            if self.parallelize_data_collection:
                if it_ == 0:  # start data collection process initially
                    print("Spawned process")
                    self.start_new_collect_data_process()
                    self.collect_data_process.start()
                    self.update_parallel_weights()  # pass in the initial weights (to start the collection process)
            else:
                sample_tasks = np.random.choice(self.train_tasks, self.num_tasks_sample, replace=False)
                print('sampled tasks', sample_tasks)
                all_rets = []
                all_final_rets = []
                # TODO: score sampled data here as a proxy for eval
                for i, idx in enumerate(sample_tasks):
                    print('task: {} / {}'.format(i, len(sample_tasks)))
                    self.task_idx = idx
                    self.env.reset_task(idx)
                    #self.enc_replay_buffer.task_buffers[idx].clear()

                    # TODO: don't hardcode max_trajs
                    # collect some trajectories with z ~ prior
                    if self.num_steps_prior > 0:
                        _, _ = self.collect_data(self.num_steps_prior, 1, np.inf, max_trajs=10)
                    # collect some trajectories with z ~ posterior
                    if self.num_steps_posterior > 0:
                        rets, final_rets = self.collect_data(self.num_steps_posterior, 1, self.update_post_train, max_trajs=10)
                        all_rets += rets
                        all_final_rets += final_rets
                    # even if encoder is trained only on samples from the prior, the policy needs to learn to handle z ~ posterior
                    if self.num_extra_rl_steps_posterior > 0:
                        rets, final_rets = self.collect_data(self.num_extra_rl_steps_posterior, 1, self.update_post_train, add_to_enc_buffer=False, max_trajs=10)
                        all_rets += rets
                        all_final_rets += final_rets

                # log returns from data collection
                avg_data_collection_returns = np.mean(all_rets)
                avg_final_collection_returns = np.mean(all_final_rets)
                self.log_data_collection_returns(avg_data_collection_returns, avg_final_collection_returns)

            print('training')
            # sample train tasks and compute gradient updates on parameters

            # update buffers
            if self.parallelize_data_collection:
                # maybe there's a better way to see if task is finished?
                while self.status_shared.value == 0:  # train in batches until we get new data
                    for train_step in range(self.num_train_steps_per_itr):  # This could be 1?
                        indices = np.random.choice(self.train_tasks, self.meta_batch)
                        self._do_training(indices)
                        self._n_train_steps_total += 1
                    gt.stamp('train')
                gt.stamp('sample')  # this is not quite correct unles num_train_steps_per_itr is 1
                self.halt_process_and_update()
            else:
                for train_step in range(self.num_train_steps_per_itr):
                    indices = np.random.choice(self.train_tasks, self.meta_batch)
                    self._do_training(indices)
                    self._n_train_steps_total += 1
                gt.stamp('train')

            for key, value in self.train_statistics.items():
                self.loggers[0].record_tabular(key, value)

            self.training_mode(False)
            self.train_statistics = None

            # eval
            started_eval = False
            if self.eval_interval > 0 and it_ % self.eval_interval == 0 and it_ != 0:
                self._try_to_eval(it_)
                gt.stamp('eval')
                started_eval = True

            # Send new weights over after evaling
            if self.parallelize_data_collection:
                self.update_parallel_weights()
            self._end_epoch(it_)
        if self.parallelize_data_collection:
            # self.collect_data_process.terminate()  # is this necessary?
            self.collect_data_process.join()

    def pretrain(self):
        """
        Do anything before the main training phase.
        """
        pass

    def log_data_collection_returns(self, avg_overall_collection_returns, avg_final_collection_returns):
        self.loggers[0].record_tabular('AvgDataCollectionReturns', avg_overall_collection_returns)
        self.loggers[0].record_tabular('AvgFinalDataCollectionReturns', avg_final_collection_returns)

    def collect_data(self, num_samples, resample_z_rate, update_posterior_rate, add_to_enc_buffer=True, max_trajs=np.inf):
        '''
        get trajectories from current env in batch mode with given policy
        collect complete trajectories until the number of collected transitions >= num_samples OR number of trajs exceeds a max (default is infinite)
        :param agent: policy to rollout
        :param num_samples: total number of transitions to sample
        :param resample_z_rate: how often to resample latent context z (in units of trajectories)
        :param update_posterior_rate: how often to update q(z | c) from which z is sampled (in units of trajectories)
        :param add_to_enc_buffer: whether to add collected data to encoder replay buffer
        '''
        # start from the prior
        self.agent.clear_z()

        num_transitions, num_trajs = 0, 0
        all_rets = []
        all_final_rets = []
        posterior_samples = False
        while num_transitions < num_samples and num_trajs < max_trajs:
            paths, n_samples = self.sampler.obtain_samples(max_samples=num_samples - num_transitions,
                                                                max_trajs=update_posterior_rate,
                                                                accum_context=False,
                                                                resample=resample_z_rate)
            num_transitions += n_samples
            num_trajs += len(paths)
            self.replay_buffer.add_paths(self.task_idx, paths)
            # compute returns on collected samples
            if posterior_samples:
                all_rets += [eval_util.get_average_returns([p]) for p in paths]
                all_final_rets += [eval_util.get_average_final_returns([p]) for p in paths]
            if add_to_enc_buffer:
                self.enc_replay_buffer.add_paths(self.task_idx, paths)
            if update_posterior_rate != np.inf:
                context = self.prepare_context(self.task_idx)
                self.agent.infer_posterior(context)
                posterior_samples = True
        self._n_env_steps_total += num_transitions
        gt.stamp('sample')
        return all_rets, all_final_rets

    def start_new_collect_data_process(self):
        self.collect_data_process = self.process_spawner.spawn_process(self.enc_replay_buffer) #, self.env)

    def update_buffer(self, buffer, idx, paths):
        buffer.add_paths(idx, paths)

    def update_parallel_weights(self):
        agent_weights = []
        nets = self.agent.networks
        for net in nets:
            agent_weights.append(net.state_dict())
        self.weight_queue.put(agent_weights)

    def halt_process_and_update(self):
        self.status_shared.value = 0
        self._n_env_steps_total = self.n_env_steps_shared.value
        while not self.buffer_queue.empty():
            try:
                # This dictionary should contain a pair of (task_idxs, paths) for each buffer being updated
                buffer_dict = self.buffer_queue.get(block=False)
            except queue.Empty:
                break

            replay_buffer_idx, replay_buffer_paths = buffer_dict[self.replay_buffer_dict_key]
            self.update_buffer(self.replay_buffer, replay_buffer_idx, replay_buffer_paths)
            if self.enc_replay_buffer_dict_key in buffer_dict:
                enc_buffer_idx, enc_buffer_paths = buffer_dict[self.enc_replay_buffer_dict_key]
                self.update_buffer(self.enc_replay_buffer, enc_buffer_idx, enc_buffer_paths)
                        # log returns from data collection
        avg_data_collection_returns = self.mean_return_shared.value
        avg_final_data_collection_returns = self.mean_final_return_shared.value
        self.log_data_collection_returns(avg_data_collection_returns, avg_final_data_collection_returns)

        # print("Start joining")
        # self.collect_data_process.terminate()  # this seems to break something?
        # self.collect_data_process.join()  # must join after queue
        # print("Done joining")


    def _try_to_eval(self, epoch, started_eval=False):
        logger = self.loggers[1]
        logger.save_extra_data(self.get_extra_data_to_save(epoch))
        if self._can_evaluate():
            self.evaluate(epoch)

            params = self.get_epoch_snapshot(epoch)
            logger.save_itr_params(epoch, params)
            table_keys = logger.get_table_key_set()
            if self._eval_old_table_keys is not None:
                print(self._eval_old_table_keys, '\n')
                print(table_keys)
                if table_keys != self._eval_old_table_keys:
                    print("Table keys cannot change from iteration to iteration. Skipping eval for now.")
                    return
            self._eval_old_table_keys = table_keys

            logger.record_tabular(
                "Number of train steps total",
                self._n_train_steps_total,
            )
            logger.record_tabular(
                "Number of env steps total",
                self._n_env_steps_total,
            )
            logger.record_tabular(
                "Number of rollouts total",
                self._n_rollouts_total,
            )

            times_itrs = gt.get_times().stamps.itrs
            train_time = times_itrs['train'][-1]
            sample_time = times_itrs['sample'][-1]
            eval_time = times_itrs['eval'][-1] if started_eval else 0
            epoch_time = train_time + sample_time + eval_time
            total_time = gt.get_times().total

            logger.record_tabular('Train Time (s)', train_time)
            logger.record_tabular('(Previous) Eval Time (s)', eval_time)
            logger.record_tabular('Sample Time (s)', sample_time)
            logger.record_tabular('Epoch Time (s)', epoch_time)
            logger.record_tabular('Total Train Time (s)', total_time)

            logger.record_tabular("Epoch", epoch)
            logger.dump_tabular(with_prefix=False, with_timestamp=False)
        else:
            logger.log("Skipping eval for now.")

    def _can_evaluate(self):
        """
        One annoying thing about the logger table is that the keys at each
        iteration need to be the exact same. So unless you can compute
        everything, skip evaluation.
        A common example for why you might want to skip evaluation is that at
        the beginning of training, you may not have enough data for a
        validation and training set.
        :return:
        """
        # eval collects its own context, so can eval any time
        return True

    def _can_train(self):
        return all([self.replay_buffer.num_steps_can_sample(idx) >= self.batch_size for idx in self.train_tasks])

    def _get_action_and_info(self, agent, observation):
        """
        Get an action to take in the environment.
        :param observation:
        :return:
        """
        agent.set_num_steps_total(self._n_env_steps_total)
        return agent.get_action(observation,)

    def _start_epoch(self, epoch):
        self._epoch_start_time = time.time()
        self._exploration_paths = []
        self._do_train_time = 0
        for logger in self.loggers:
            logger.push_prefix('Iteration #%d | ' % epoch)

    def _end_epoch(self, epoch):
        # dump training stats to the train log
        logger = self.loggers[0]
        logger.save_extra_data(self.get_extra_data_to_save(epoch))

        params = self.get_epoch_snapshot(epoch)
        logger.save_itr_params(epoch, params)
        table_keys = logger.get_table_key_set()
        if self._old_table_keys is not None:
            assert table_keys == self._old_table_keys, (
                "Table keys cannot change from iteration to iteration."
            )
        self._old_table_keys = table_keys

        logger.record_tabular(
            "Number of train steps total",
            self._n_train_steps_total,
        )
        logger.record_tabular(
            "Number of env steps total",
            self._n_env_steps_total,
        )
        logger.record_tabular(
            "Number of rollouts total",
            self._n_rollouts_total,
        )

        times_itrs = gt.get_times().stamps.itrs
        train_time = times_itrs['train'][-1]
        sample_time = times_itrs['sample'][-1]
        epoch_time = train_time + sample_time
        total_time = gt.get_times().total

        logger.record_tabular('Train Time (s)', train_time)
        logger.record_tabular('Sample Time (s)', sample_time)
        logger.record_tabular('Epoch Time (s)', epoch_time)
        logger.record_tabular('Total Train Time (s)', total_time)

        logger.record_tabular("Epoch", epoch)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)
        logger.log("Epoch Duration: {0}".format(
            time.time() - self._epoch_start_time
        ))
        logger.log("Started Training: {0}".format(self._can_train()))
        logger.pop_prefix()

    ##### Snapshotting utils #####
    def get_epoch_snapshot(self, epoch):
        data_to_save = dict(
            epoch=epoch,
            exploration_policy=self.exploration_policy,
        )
        if self.save_environment:
            data_to_save['env'] = self.training_env
        return data_to_save

    def get_extra_data_to_save(self, epoch):
        """
        Save things that shouldn't be saved every snapshot but rather
        overwritten every time.
        :param epoch:
        :return:
        """
        if self.render:
            self.training_env.render(close=True)
        data_to_save = dict(
            epoch=epoch,
        )
        if self.save_environment:
            data_to_save['env'] = self.training_env
        if self.save_replay_buffer:
            data_to_save['replay_buffer'] = self.replay_buffer
            data_to_save['enc_replay_buffer'] = self.enc_replay_buffer
        if self.save_algorithm:
            data_to_save['algorithm'] = self
        return data_to_save

    def collect_paths(self, idx, epoch, run):
        self.task_idx = idx
        self.env.reset_task(idx)

        self.agent.clear_z()
        paths = []
        num_transitions = 0
        num_trajs = 0
        while num_transitions < self.num_steps_per_eval:
            path, num = self.sampler.obtain_samples(deterministic=self.eval_deterministic, max_samples=self.num_steps_per_eval - num_transitions, max_trajs=1, accum_context=True)
            paths += path
            num_transitions += num
            num_trajs += 1
            if num_trajs >= self.num_exp_traj_eval:
                self.agent.infer_posterior(self.agent.context)

        if self.sparse_rewards:
            for p in paths:
                sparse_rewards = np.stack(e['sparse_reward'] for e in p['env_infos']).reshape(-1, 1)
                p['rewards'] = sparse_rewards

        goal = self.env.get_goal()
        for path in paths:
            path['goal'] = goal # goal

        # save the paths for visualization, only useful for point mass
        if self.dump_eval_paths and epoch % 5 == 0:
            self.loggers[1].save_extra_data(paths, path='eval_trajectories/task{}-epoch{}-run{}'.format(idx, epoch, run))

        return paths

    def _do_eval(self, indices, epoch):
        final_returns = []
        online_returns = []
        for idx in indices:
            print('task: {}'.format(idx))
            all_rets = []
            for r in range(self.num_evals):
                paths = self.collect_paths(idx, epoch, r)
                all_rets.append([eval_util.get_average_returns([p]) for p in paths])
            final_returns.append(np.mean([a[-1] for a in all_rets]))
            # record online returns for the first n trajectories
            n = min([len(a) for a in all_rets])
            all_rets = [a[:n] for a in all_rets]
            all_rets = np.mean(np.stack(all_rets), axis=0)
            online_returns.append(all_rets)
        n = min([len(t) for t in online_returns])
        online_returns = [t[:n] for t in online_returns]
        return final_returns, online_returns

    def evaluate(self, epoch):
        if self.eval_statistics is None:
            self.eval_statistics = OrderedDict()

        ### train tasks
        # eval on a subset of train tasks for speed
        indices = np.random.choice(self.train_tasks, self.num_task_eval, replace=False)
        print('evaluating on train tasks: {}'.format(indices))
        # TODO: consider not doing this, if we can esimate overfitting from sim?
        ### eval train tasks with posterior sampled from the training replay buffer
        print('evaluating with context from replay buffer')
        train_returns = []
        for idx in indices:
            print('task: {}'.format(idx))
            self.task_idx = idx
            self.env.reset_task(idx)
            paths = []
            for _ in range(self.num_steps_per_eval // self.max_path_length):
                context = self.prepare_context(idx)
                self.agent.infer_posterior(context)
                p, _ = self.sampler.obtain_samples(deterministic=self.eval_deterministic, max_samples=self.max_path_length,
                                                        accum_context=False,
                                                        max_trajs=1,
                                                        resample=np.inf)
                paths += p

            if self.sparse_rewards:
                for p in paths:
                    sparse_rewards = np.stack(e['sparse_reward'] for e in p['env_infos']).reshape(-1, 1)
                    p['rewards'] = sparse_rewards

            train_returns.append(eval_util.get_average_returns(paths))
        train_returns = np.mean(train_returns)
        print('evaluating with on-policy context')
        ### eval train tasks with on-policy data to match eval of test tasks
        train_final_returns, train_online_returns = self._do_eval(indices, epoch)
        eval_util.dprint('train online returns')
        eval_util.dprint(train_online_returns)

        ### test tasks
        indices = np.random.choice(self.eval_tasks, self.num_task_eval, replace=False)
        print('evaluating on test tasks: {}'.format(indices))
        test_final_returns, test_online_returns = self._do_eval(indices, epoch)
        eval_util.dprint('test online returns')
        eval_util.dprint(test_online_returns)

        # save the final posterior
        self.agent.log_diagnostics(self.eval_statistics)

        if hasattr(self.env, "log_diagnostics"):
            self.env.log_diagnostics(paths)

        avg_train_return = np.mean(train_final_returns)
        avg_test_return = np.mean(test_final_returns)
        avg_train_online_return = np.mean(np.stack(train_online_returns), axis=0)
        avg_test_online_return = np.mean(np.stack(test_online_returns), axis=0)
        self.eval_statistics['AverageOverallTrainReturn_all_train_tasks'] = train_returns
        self.eval_statistics['AverageFinalReturn_all_train_tasks'] = avg_train_return
        self.eval_statistics['AverageFinalReturn_all_test_tasks'] = avg_test_return
        self.loggers[1].save_extra_data(avg_train_online_return, path='online-train-epoch{}'.format(epoch))
        self.loggers[1].save_extra_data(avg_test_online_return, path='online-test-epoch{}'.format(epoch))

        for key, value in self.eval_statistics.items():
            self.loggers[1].record_tabular(key, value)
        self.eval_statistics = None

        if self.render_eval_paths:
            self.env.render_paths(paths)

        if self.plotter:
            self.plotter.draw()

    @abc.abstractmethod
    def training_mode(self, mode):
        """
        Set training mode to `mode`.
        :param mode: If True, training will happen (e.g. set the dropout
        probabilities to not all ones).
        """
        pass

    @abc.abstractmethod
    def _do_training(self):
        """
        Perform some update, e.g. perform one gradient step.
        :return:
        """
        pass
