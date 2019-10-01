class ProcessSpawner:
    def __init__(self, buffer_queue, train_tasks, max_path_length, num_tasks_sample, num_steps_prior, num_steps_posterior,
                 num_extra_rl_steps_posterior, update_post_train, replay_buffer_dict_key, enc_replay_buffer_dict_key,
                 embedding_batch_size, recurrent):
        self.buffer_queue = buffer_queue
        self.train_tasks = train_tasks
        self.max_path_length = max_path_length
        self.num_tasks_sample = num_tasks_sample
        self.num_steps_prior = num_steps_prior
        self.num_steps_posterior = num_steps_posterior
        self.num_extra_rl_steps_posterior = num_extra_rl_steps_posterior
        self.update_post_train = update_post_train
        self.replay_buffer_dict_key = replay_buffer_dict_key
        self.enc_replay_buffer_dict_key = enc_replay_buffer_dict_key
        self.embedding_batch_size = embedding_batch_size
        self.recurrent = recurrent

    def spawn_process(self, enc_replay_buffer, env, agent_weights, n_env_steps_shared, mean_return_shared, is_done_shared):
        return mp.Process(target=self.collect_data_routine, args=(self.buffer_queue, enc_replay_buffer, env, agent_weights, n_env_steps_shared, mean_return_shared, is_done_shared))

    def collect_data_routine(self, buffer_queue, enc_replay_buffer, env, agent_weights, n_env_steps_shared, mean_return_shared, is_done_shared):
        context_encoder = encoder_model(
            hidden_sizes=[200, 200, 200],
            input_size=obs_dim + action_dim + reward_dim,
            output_size=context_encoder,
        )
        policy = TanhGaussianPolicy(
            hidden_sizes=[net_size, net_size, net_size],
            obs_dim=obs_dim + latent_dim,
            latent_dim=latent_dim,
            action_dim=action_dim,
        )
        context_encoder.load_state_dict(agent_weights[0])
        policy.load_state_dict(agent_weights[1])
        agent = PEARLAgent(
            latent_dim,
            context_encoder,
            policy,
            **variant['algo_params']
        )

        sampler = InPlacePathSampler(
            env=env,
            policy=agent,
            max_path_length=self.max_path_length,
        )

        sample_tasks = np.random.choice(self.train_tasks, self.num_tasks_sample, replace=False)
        print('sampled tasks', sample_tasks)
        all_rets = []
        # TODO: score sampled data here as a proxy for eval
        for i, idx in enumerate(sample_tasks):
            print('task: {} / {}'.format(i, len(sample_tasks)))
            task_idx = idx
            env.reset_task(idx)
            # self.enc_replay_buffer.task_buffers[idx].clear()

            # TODO: don't hardcode max_trajs
            # collect some trajectories with z ~ prior
            if self.num_steps_prior > 0:
                _ = self.collect_data_parallel(task_idx, buffer_queue, enc_replay_buffer, n_env_steps_shared, sampler,
                                               agent, self.num_steps_prior, 1, np.inf, max_trajs=10)
            # collect some trajectories with z ~ posterior
            if self.num_steps_posterior > 0:
                rets = self.collect_data_parallel(task_idx, buffer_queue, enc_replay_buffer, n_env_steps_shared,
                                                  sampler, agent, self.num_steps_posterior, 1, self.update_post_train,
                                                  max_trajs=10)
                all_rets += rets
            # even if encoder is trained only on samples from the prior, the policy needs to learn to handle z ~ posterior
            if self.num_extra_rl_steps_posterior > 0:
                rets = self.collect_data_parallel(task_idx, buffer_queue, enc_replay_buffer, n_env_steps_shared,
                                                  sampler, agent, self.num_extra_rl_steps_posterior, 1,
                                                  self.update_post_train, add_to_enc_buffer=False, max_trajs=10)
                all_rets += rets
        print("FINISHED COLLECTING DATA")
        is_done_shared.value = 1
        mean_return_shared.value = np.mean(all_rets)

    def collect_data_parallel(self, task_idx, buffer_queue, enc_replay_buffer, n_env_steps_shared, sampler, agent,
                              num_samples, resample_z_rate, update_posterior_rate, add_to_enc_buffer=True, max_trajs=np.inf):
        '''
        get trajectories from current env in batch mode with given policy
        collect complete trajectories until the number of collected transitions >= num_samples OR number of trajs exceeds a max (default is infinite)

        :param agent: policy to rollout
        :param num_samples: total number of transitions to sample
        :param resample_z_rate: how often to resample latent context z (in units of trajectories)
        :param update_posterior_rate: how often to update q(z | c) from which z is sampled (in units of trajectories)
        :param add_to_enc_buffer: whether to add collected data to encoder replay buffer
        '''
        buffer_dict = {}

        # start from the prior
        print("AGENT: " + str(agent))
        agent.clear_z()

        num_transitions, num_trajs = 0, 0
        all_rets = []
        posterior_samples = False
        while num_transitions < num_samples and num_trajs < max_trajs:
            paths, n_samples = sampler.obtain_samples(max_samples=num_samples - num_transitions,
                                                      max_trajs=update_posterior_rate,
                                                      accum_context=False,
                                                      resample=resample_z_rate)
            num_transitions += n_samples
            num_trajs += len(paths)
            print("ADDED TO REP BUFFER: " + str(len(paths)))
            buffer_dict[self.replay_buffer_dict_key] = (task_idx, paths)
            # compute returns on collected samples
            if posterior_samples:
                all_rets += [eval_util.get_average_returns([p]) for p in paths]
            if add_to_enc_buffer:
                print("ADDED TO ENC BUFFER: " + str(len(paths)))
                buffer_dict[self.enc_replay_buffer_dict_key] = (task_idx, paths)
            if update_posterior_rate != np.inf:
                context = self.prepare_context(task_idx, enc_replay_buffer)
                agent.infer_posterior(context)
                posterior_samples = True
            buffer_queue.put(buffer_dict)
        with n_env_steps_shared.get_lock():
            n_env_steps_shared.value += num_transitions
        # gt.stamp('sample')  # TODO: make stamping work for parallel?
        return all_rets

    def prepare_encoder_data(self, obs, act, rewards):
        ''' prepare context for encoding '''
        # for now we embed only observations and rewards
        # assume obs and rewards are (task, batch, feat)
        task_data = torch.cat([obs, act, rewards], dim=2)
        return task_data

    def prepare_context(self, idx, enc_replay_buffer):
        ''' sample context from replay buffer and prepare it '''
        batch = ptu.np_to_pytorch_batch(enc_replay_buffer.random_batch(idx, batch_size=self.embedding_batch_size, sequence=self.recurrent))
        obs = batch['observations'][None, ...]
        act = batch['actions'][None, ...]
        rewards = batch['rewards'][None, ...]
        context = self.prepare_encoder_data(obs, act, rewards)
        return context
