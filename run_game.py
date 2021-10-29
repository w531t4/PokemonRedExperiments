import argparse
from multiprocessing import Process

from red_game import RedEnv
from random_agent import RandomAgent

def run_random_sims(pid):
    sim = RedEnv(headless=True)
    em_id = sim.instance_id
    print(f'Initialized emulator {em_id}')
    rand_agent = RandomAgent()

    count = 0
    total_reward = 0
    max_reward = 0
    min_reward = 999999
    while True:
        reward = sim.play_episode(rand_agent, 10000)
        count += 1
        total_reward += reward
        min_reward = min(min_reward, reward)
        max_reward = max(max_reward, reward)
        print(f'{em_id} round {count}, reward: {reward}, min: {min_reward}, max: {max_reward}, average: {total_reward/count}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run game options')
    parser.add_argument('-p', dest='procs', type=int, default=1, help='Number of games to run in parallel')
    args = parser.parse_args()
    print(f'Running {args.procs} emulator{"s in parallel" if args.procs > 1 else ""}')
    processes = [Process(target=run_random_sims, args=(i,)) for i in range(args.procs)]
    for p in processes:
        p.daemon = True
        p.start()
    for p in processes:
        p.join()