def train(env, agent, encoder, epochs=1000):
    buffer = RolloutBuffer()

    for ep in range(epochs):
        state = env.reset()
        done = False

        while not done:
            encoded_state = encoder.encode(env, env.current_player)
            mask = env.get_action_mask()

            action, logprob, value = agent.act(encoded_state, mask)
            _, done, winner = env.step(action)

            reward = compute_reward(env, env.current_player, done, winner)

            buffer.states.append(encoded_state)
            buffer.actions.append(action)
            buffer.logprobs.append(logprob)
            buffer.rewards.append(reward)
            buffer.values.append(value)
            buffer.dones.append(done)

        # 👉 PPO update ở đây
