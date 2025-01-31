import time

from .agent import NetworkConfig, get_network, AgentConfig
from .environment import RnaDesignEnvironment, RnaDesignEnvironmentConfig
from tensorforce.environments import Environment
from tensorforce.agents import Agent
from tensorforce.execution import Runner
from tqdm import tqdm
import os
import pickle

class TransformerLearna:
    def run(
        self,
        dot_brackets,
        episode_timeout=300,
        network_config=NetworkConfig(),
        agent_config=AgentConfig(),
        env_config=RnaDesignEnvironmentConfig(),
        num_epochs=1,
        agent_dir="",
        agent_filename=""
    ):
        """
        Main function for RNA design. Instantiate an environment and an agent to run in a
        tensorforce runner.

        Args:
            timeout: Maximum time to run.
            restore_path: Path to restore saved configurations/models from.
            stop_learning: If set, no weight updates are performed (Meta-LEARNA).
            restart_timeout: Time interval for restarting of the agent.
            network_config: The configuration of the network.
            agent_config: The configuration of the agent.
            env_config: The configuration of the environment.

        Returns:
            Results
        """

        env_config.use_embedding = bool(network_config.embedding_dim)
        
        environment = Environment.create(
            environment=RnaDesignEnvironment, max_episode_timesteps=episode_timeout * 20, dot_brackets=dot_brackets, env_config=env_config
        )

        network = get_network(network_config)

        if os.path.exists(f"{agent_dir}/{agent_filename}"):
            agent = Agent.load(directory=agent_dir, agent_filename=agent_filename, environment=environment, **agent_config.__dict__)
        else:
            agent = Agent.create(agent='ppo', network=network, environment=environment, **agent_config.__dict__)

        results = dict()


        for epoch in range(num_epochs):
            print(f"Epoch: {epoch}")
            for _ in tqdm(range(len(dot_brackets))):
                states = environment.reset()
                episode_start_time = time.time()
                terminal = False  # Initialize terminal flag
                internals = agent.initial_internals()
                while not terminal:
                    # Agent acts based on the current state
                    actions, internals = agent.act(states=states, internals=internals)

                    # Execute actions in the environment
                    next_states, terminal, reward = environment.execute(actions=actions)

                    # Check for episode timeout
                    elapsed_time = time.time() - episode_start_time
                    if elapsed_time >= episode_timeout:
                        terminal = True

                    # Observe the result of the action
                    agent.observe(terminal=terminal, reward=reward)

                    # Update states
                    states = next_states

                    # Check if the environment has solved the problem
                    if environment.design.all_bases_once_solved:
                        candidate_solution = environment.design.primary
                        structure = environment.target.dot_bracket
                        candidate = (candidate_solution, reward)
                        current_best_reward = results.get(structure, candidate)[1]
                        if reward >= current_best_reward:
                            results[structure] = candidate

                if agent_dir and agent_filename:
                    agent.save(directory=agent_dir, filename=agent_filename, internals=internals)

        return {structure: result[0] for structure, result in results.items()}