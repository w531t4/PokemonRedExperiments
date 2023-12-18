
import sys
import uuid
import os
from math import floor, sqrt
import json
from pathlib import Path
import hnswlib
import numpy as np

from einops import rearrange
import matplotlib.pyplot as plt
from skimage.transform import resize
from pyboy import PyBoy
from pyboy.logger import log_level
import mediapy as media
import pandas as pd
from typing import TypedDict, Dict, Union, List, Tuple, Any, NamedTuple

from gymnasium import Env, spaces
from pyboy.utils import WindowEvent

class RGEnvConfig(TypedDict):
    debug: bool
    headless: bool
    save_final_state: bool
    action_freq: int
    init_state: str
    max_steps: int
    print_rewards: bool
    save_video: bool
    fast_video: bool
    session_path: Path
    gb_path: str
    sim_frame_dist: float
    use_screen_explore: bool
    reward_scale: int
    extra_buttons: bool
    explore_weight: int

class MapLocation(NamedTuple):
    x: int
    y: int
    map_id: int
    map_name: str

class RedGymEnv(Env):
    def __init__(self,
                 config: RGEnvConfig = None,
                 ) -> None:
        self.debug = config['debug']
        self.s_path = config['session_path']
        self.save_final_state = config['save_final_state']
        self.print_rewards = config['print_rewards']
        self.vec_dim = 4320 #1000
        self.headless = config['headless']
        self.num_elements = 20000 # max
        self.init_state = config['init_state']
        self.act_freq = config['action_freq']
        self.max_steps = config['max_steps']
        self.early_stopping = config['early_stop']
        self.save_video = config['save_video']
        self.fast_video = config['fast_video']
        self.video_interval = 256 * self.act_freq
        self.downsample_factor = 2
        self.frame_stacks = 3
        self.explore_weight = 1 if 'explore_weight' not in config else config['explore_weight']
        self.use_screen_explore = True if 'use_screen_explore' not in config else config['use_screen_explore']
        self.similar_frame_dist = config['sim_frame_dist']
        self.reward_scale = 1 if 'reward_scale' not in config else config['reward_scale']
        self.extra_buttons = False if 'extra_buttons' not in config else config['extra_buttons']
        self.instance_id = str(uuid.uuid4())[:8] if 'instance_id' not in config else config['instance_id']
        self.s_path.mkdir(exist_ok=True)
        self.reset_count = 0
        self.all_runs: List[Dict[str, Union[int, float]]] = []

        self.did_knn_count_change: bool = False
        # Set this in SOME subclasses
        self.metadata = {"render.modes": []}
        self.reward_range = (0, 15000)

        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
        ]

        if self.extra_buttons:
            self.valid_actions.extend([
                WindowEvent.PRESS_BUTTON_START,
                WindowEvent.PASS
            ])

        self.release_arrow = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP
        ]

        self.release_button = [
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B
        ]

        self.output_shape = (36, 40, 3)
        self.mem_padding = 2
        self.memory_height = 8
        self.col_steps = 16
        self.output_full = (self.output_shape[0] * self.frame_stacks + 2 * (self.mem_padding + self.memory_height),
                            self.output_shape[1],
                            self.output_shape[2],
        )

        # Set these in ALL subclasses
        self.action_space = spaces.Discrete(len(self.valid_actions))
        self.observation_space = spaces.Box(low=0,
                                            high=255,
                                            shape=self.output_full,
                                            dtype=np.uint8,
                                            )

        head = 'headless' if config['headless'] else 'SDL2'

        log_level("ERROR")
        self.pyboy = PyBoy(config['gb_path'],
                           debugging=False,
                           disable_input=False,
                           window_type=head,
                           hide_window='--quiet' in sys.argv,
            )

        self.seen_coords: Dict[str, int] = {}
        self.screen = self.pyboy.botsupport_manager().screen()

        if not config['headless']:
            self.pyboy.set_emulation_speed(6)
        else:
            self.pyboy.set_emulation_speed(0)
        self.reset()

    def reset(self,
              seed: Any = None,
              ) -> Tuple[np.ndarray, dict]:
        self.seed = seed
        # restart game, skipping credits
        with open(self.init_state, "rb") as f:
            self.pyboy.load_state(f)

        if self.use_screen_explore:
            self.init_knn()
        else:
            self.seen_coords = {}

        self.recent_memory = np.zeros((self.output_shape[1]*self.memory_height,
                                       3,
                                       ),
                                      dtype=np.uint8,
                                      )

        self.recent_frames = np.zeros((self.frame_stacks,
                                       self.output_shape[0],
                                       self.output_shape[1],
                                       self.output_shape[2],
                                       ),
                                      dtype=np.uint8,
                                      )

        self.agent_stats: List[Dict[str, Union[int, str, List[int], float]]] = []

        if self.save_video:
            base_dir = self.s_path / Path('rollouts')
            base_dir.mkdir(exist_ok=True)
            full_name = Path(f'full_reset_{self.reset_count}_id{self.instance_id}').with_suffix('.mp4')
            model_name = Path(f'model_reset_{self.reset_count}_id{self.instance_id}').with_suffix('.mp4')
            self.full_frame_writer = media.VideoWriter(base_dir / full_name, (144, 160), fps=60)
            self.full_frame_writer.__enter__()
            self.model_frame_writer = media.VideoWriter(base_dir / model_name, self.output_full[:2], fps=60)
            self.model_frame_writer.__enter__()

        self.levels_satisfied = False
        self.base_explore = 0
        self.max_opponent_level = 0
        self.max_event_rew = 0
        self.max_level_rew = 0
        self.total_healing_rew = 0
        self.died_count = 0
        self.step_count = 0
        self.progress_reward = self.get_game_state_reward()
        self.total_reward = sum([val for _, val in self.progress_reward.items()])
        self.reset_count += 1
        self.did_knn_count_change = False
        return self.render(), {}

    def init_knn(self) -> None:
        # Declaring index
        self.knn_index = hnswlib.Index(space='l2',
                                       dim=self.vec_dim,
                                       ) # possible options are l2, cosine or ip
        # Initing index - the maximum number of elements should be known beforehand
        self.knn_index.init_index(max_elements=self.num_elements,
                                  ef_construction=100,
                                  M=16,
                                  )

    def render(self,
               reduce_res: bool = True,
               add_memory: bool = True,
               update_mem: bool = True,
               ) -> np.ndarray:
        game_pixels_render = self.screen.screen_ndarray() # (144, 160, 3)
        if reduce_res:
            game_pixels_render = (255*resize(game_pixels_render,
                                             self.output_shape)).astype(np.uint8)
            if update_mem:
                self.recent_frames[0] = game_pixels_render
            if add_memory:
                pad = np.zeros(shape=(self.mem_padding,
                                      self.output_shape[1],
                                      3,
                                     ),
                               dtype=np.uint8,
                               )
                game_pixels_render = np.concatenate((self.create_exploration_memory(),
                                                     pad,
                                                     self.create_recent_memory(),
                                                     pad,
                                                     rearrange(self.recent_frames,
                                                               'f h w c -> (f h) w c',
                                                               ),
                                                    ),
                                                    axis=0,
                                                    )
        return game_pixels_render

    def step(self,
             action: int,
             ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # Observations needed PRIOR to executing the step
        before_health = self.read_hp_fraction()
        before_party_size = self.read_m(0xD163)
        before_prog = self.group_rewards(hp=before_health)
        # The ACTION
        self.run_action_on_emulator(action)

        location = self.get_current_location()
        after_health = self.read_hp_fraction()
        after_party_size = self.read_m(0xD163)
        # The screen transitions (prior to battle) seem to be encouraging
        # it to go into battle more often. 0xd57 is non-zero when in battle
        # (and during transitions).
        battle_indicator = self.read_m(0xD057)

        self.append_agent_stats(action,
                                location=location,
                                party_size=after_party_size,
                                )

        self.recent_frames = np.roll(self.recent_frames,
                                     1,
                                     axis=0,
                                     )
        obs_memory = self.render()

        # trim off memory from frame for knn index
        frame_start = 2 * (self.memory_height + self.mem_padding)
        obs_flat: np.float32 = obs_memory[frame_start:frame_start+self.output_shape[0],
                              ...].flatten().astype(np.float32)


        curr_knn_count = str(round(self.knn_index.get_current_count(), 5))
        if battle_indicator == 0:
            if self.use_screen_explore:
                self.update_frame_knn_index(obs_flat)
            else:
                self.update_seen_coords(location=location)
        self.did_knn_count_change = str(round(self.knn_index.get_current_count(), 5)) != curr_knn_count
        self.update_heal_reward(current_health=after_health,
                                last_health=before_health,
                                before_party_size=before_party_size,
                                after_party_size=after_party_size,
                                )

        # compute reward
        self.progress_reward = self.get_game_state_reward()
        new_reward, new_prog = self.update_reward(old_prog=before_prog,
                                                  hp=after_health,
                                                  )

        # shift over short term reward memory
        self.recent_memory = np.roll(self.recent_memory, 3)
        self.recent_memory[0, 0] = min(new_prog[0] * 64, 255)
        self.recent_memory[0, 1] = min(new_prog[1] * 64, 255)
        self.recent_memory[0, 2] = min(new_prog[2] * 128, 255)

        step_limit_reached = self.check_if_done()

        self.save_and_print_info(step_limit_reached,
                                 obs_memory,
                                 location=location,
                                 battle_indicator=battle_indicator,
                                 )

        self.step_count += 1

        return (obs_memory,
                new_reward*0.1,
                False,
                step_limit_reached,
                {},
        )

    def run_action_on_emulator(self, action: int) -> None:
        # press button then release after some steps
        if action == WindowEvent.PASS:
            self.pyboy.send_input(WindowEvent.PASS)
        else:
            self.pyboy.send_input(self.valid_actions[action])
        # disable rendering when we don't need it
        if not self.save_video and self.headless:
            self.pyboy._rendering(False)
        for i in range(self.act_freq):
            # release action, so they are stateless
            if i == 8:
                if action < 4:
                    # release arrow
                    self.pyboy.send_input(self.release_arrow[action])
                elif action > 3 and action < 6:
                    # release button
                    self.pyboy.send_input(self.release_button[action - 4])
                elif action == WindowEvent.PASS:
                    pass
                elif self.valid_actions[action] == WindowEvent.PRESS_BUTTON_START:
                    self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)
            if self.save_video and not self.fast_video:
                self.add_video_frame()
            if i == self.act_freq-1:
                self.pyboy._rendering(True)
            self.pyboy.tick()
        if self.save_video and self.fast_video:
            self.add_video_frame()

    def add_video_frame(self) -> None:
        self.full_frame_writer.add_image(self.render(reduce_res=False,
                                                     update_mem=False,
                                                     ))
        self.model_frame_writer.add_image(self.render(reduce_res=True,
                                                      update_mem=False,
                                                      ))

    def get_current_location(self) -> MapLocation:
        map_id = self.read_m(0xD35E)
        return MapLocation(x=self.read_m(0xD362),
                           y=self.read_m(0xD361),
                           map_id=map_id,
                           map_name=self.__class__.get_map_location(map_id),
                           )

    def append_agent_stats(self,
                           action: int,
                           location: MapLocation,
                           party_size: int,
                           ) -> None:
        levels = self.get_my_pokemon_levels()

        expl: Tuple[str, int]
        if self.use_screen_explore:
            expl = ('frames',
                    # self.khn_index.get_current_count returns int
                    self.knn_index.get_current_count())
        else:
            expl = ('coord_count',
                    len(self.seen_coords))
        self.agent_stats.append({'step': self.step_count,
                                 'x': location.x,
                                 'y': location.y,
                                 'map': location.map_id,
                                 'map_location': location.map_name,
                                 'last_action': action,
                                 'pcount': party_size,
                                 'levels': levels,
                                 'levels_sum': sum(levels),
                                 'ptypes': self.read_party(),
                                 'hp': self.read_hp_fraction(),
                                 expl[0]: expl[1],
                                 'deaths': self.died_count,
                                 'badge': self.get_badges(),
                                 'event': self.progress_reward['event'],
                                 'healr': self.total_healing_rew,
        })

    def update_frame_knn_index(self,
                               frame_vec: np.float32,
                               ) -> None:

        if self.get_levels_sum() >= 22 and not self.levels_satisfied:
            self.levels_satisfied = True
            # self.khn_index.get_current_count returns int
            self.base_explore = self.knn_index.get_current_count()
            self.init_knn()

        # self.khn_index.get_current_count returns int
        if self.knn_index.get_current_count() == 0:
            # if index is empty add current frame
            self.knn_index.add_items(# self.khn_index.get_current_count returns int
                                     frame_vec,
                                     np.array([self.knn_index.get_current_count()]),
            )
        else:
            # check for nearest frame and add if current
            labels, distances = self.knn_index.knn_query(frame_vec,
                                                         k = 1,
                                                         )
            if distances[0][0] > self.similar_frame_dist:
                # print(f"distances[0][0] : {distances[0][0]} similar_frame_dist : {self.similar_frame_dist}")
                self.knn_index.add_items(
                    # self.khn_index.get_current_count returns int
                    frame_vec, np.array([self.knn_index.get_current_count()])
                )

    def update_seen_coords(self,
                           location: MapLocation,
                           ) -> None:
        coord_string = f"x:{location.x} y:{location.y} m:{location.map_id}"
        if self.get_levels_sum() >= 22 and not self.levels_satisfied:
            self.levels_satisfied = True
            self.base_explore = len(self.seen_coords)

        self.seen_coords[coord_string] = self.step_count

    @staticmethod
    def get_reward_delta(old: Tuple[float, float, float],
                         new: Tuple[float, float, float],
                         ) -> Tuple[float, float, float]:
        return (new[0]-old[0],
                new[1]-old[1],
                new[2]-old[2],
                )

    def update_reward(self,
                      old_prog: Tuple[float, float, float],
                      hp: float,
                      ) -> Tuple[Union[int, float],
                                 Tuple[float, float, float]]:
        new_prog = self.group_rewards(hp=hp)
        new_total = sum([val for _, val in self.progress_reward.items()]) #sqrt(self.explore_reward * self.progress_reward)
        new_step = new_total - self.total_reward
        if new_step < 0 and hp > 0:
            #print(f'\n\nreward went down! {self.progress_reward}\n\n')
            self.save_screenshot('neg_reward')

        self.total_reward = new_total
        delta = self.get_reward_delta(old=old_prog,
                                      new=new_prog,
                                      )
        return (new_step, delta)

    def group_rewards(self,
                      hp: float,
                      ) -> Tuple[float, float, float]:
        prog = self.progress_reward
        # these values are only used by memory
        return (prog['level'] * 100 / self.reward_scale,
                hp*2000,
                prog['explore'] * 150 / (self.explore_weight * self.reward_scale))
               #(prog['events'],
               # prog['levels'] + prog['party_xp'],
               # prog['explore'])

    def create_exploration_memory(self) -> np.ndarray:
        w = self.output_shape[1]
        h = self.memory_height

        def make_reward_channel(r_val: float) -> np.ndarray:
            col_steps = self.col_steps
            max_r_val = (w-1) * h * col_steps
            # truncate progress bar. if hitting this
            # you should scale down the reward in group_rewards!
            r_val = min(r_val,
                        max_r_val,
                        )
            row = floor(r_val / (h * col_steps))
            memory = np.zeros(shape=(h, w),
                              dtype=np.uint8,
                              )
            memory[:, :row] = 255
            row_covered = row * h * col_steps
            col = floor((r_val - row_covered) / col_steps)
            memory[:col, row] = 255
            col_covered = col * col_steps
            last_pixel = floor(r_val - row_covered - col_covered)
            memory[col, row] = last_pixel * (255 // col_steps)
            return memory

        level, hp, explore = self.group_rewards(hp=self.read_hp_fraction())
        full_memory = np.stack((make_reward_channel(level),
                                make_reward_channel(hp),
                                make_reward_channel(explore)
                                ),
                               axis=-1,
                               )

        if self.get_badges() > 0:
            full_memory[:, -1, :] = 255

        return full_memory

    def create_recent_memory(self) -> np.ndarray:
        return rearrange(self.recent_memory,
                         '(w h) c -> h w c',
                         h=self.memory_height,
                         )

    def check_if_done(self) -> bool:
        if self.early_stopping:
            done = False
            if self.step_count > 128 and self.recent_memory.sum() < (255 * 1):
                done = True
        else:
            done = self.step_count >= self.max_steps
        #done = self.read_hp_fraction() == 0
        return done

    def save_and_print_info(self,
                            done: bool,
                            obs_memory: np.ndarray,
                            location: MapLocation,
                            battle_indicator: int,
                            status_interval: int = 20,
                            ) -> None:
        if self.print_rewards and (self.step_count % status_interval) == 0:
            out_data: Dict[str, str] = dict()
            out_data["step"] = self.step_count # f'{self.step_count:6d}'
            out_data["location"] = location.map_name
            for key, val in self.progress_reward.items():
                out_data[key] = round(val, 2) # f' {key}: {val:5.2f}'
            out_data["sum"] = round(self.total_reward, 2) # f'{self.total_reward:5.2f}'
            out_data["lvlsum"] = self.get_levels_sum()
            out_data["i_id"] = self.instance_id
            out_data["battle_status"] = battle_indicator
            out_data["knn_chg"] = self.did_knn_count_change
            json_string = json.dumps(out_data)
            print('\r%s' % json_string,
                  end='',
                  flush=True,
                  )
            txt_status = self.s_path / Path('txt_status')
            txt_status.mkdir(exist_ok=True)
            txt_total = txt_status / str(self.instance_id)
            txt_total.write_text("%s\n" % json_string)

        if self.step_count % status_interval == 0:
            plt.imsave(self.s_path / Path(f'curframe_{self.instance_id}.jpeg'),
                       self.render(reduce_res=False),
                       )

        if self.print_rewards and done:
            print('', flush=True)
            if self.save_final_state:
                fs_path = self.s_path / Path('final_states')
                fs_path.mkdir(exist_ok=True)
                plt.imsave(fs_path / Path(f'frame_r{self.total_reward:.4f}_{self.reset_count}_small.jpeg'),
                           obs_memory,
                           )
                plt.imsave(fs_path / Path(f'frame_r{self.total_reward:.4f}_{self.reset_count}_full.jpeg'),
                           self.render(reduce_res=False),
                           )

        if self.save_video and done:
            self.full_frame_writer.close()
            self.model_frame_writer.close()

        if done:
            self.all_runs.append(self.progress_reward)
            with open(self.s_path / Path(f'all_runs_{self.instance_id}.json'), 'w') as f:
                json.dump(self.all_runs, f)
            pd.DataFrame(self.agent_stats).to_csv(self.s_path / Path(f'agent_stats_{self.instance_id}.csv.gz'),
                                                  compression='gzip',
                                                  mode='a',
                                                  )

    def read_m(self,
               addr: int,
               ) -> int:
        return self.pyboy.get_memory_value(addr)

    def read_bit(self,
                 addr,
                 bit: int,
                 ) -> bool:
        # add padding so zero will read '0b100000000' instead of '0b0'
        return bin(256 + self.read_m(addr))[-bit-1] == '1'

    def get_my_pokemon_levels(self) -> List[int]:
        return [self.read_m(x) for x in [0xD18C,
                                         0xD1B8,
                                         0xD1E4,
                                         0xD210,
                                         0xD23C,
                                         0xD268,
                                         ]]
    def get_levels_sum(self):
        poke_levels = [max(x - 2, 0) for x in self.get_my_pokemon_levels()]
        return max(sum(poke_levels) - 4, 0) # subtract starting pokemon level

    def get_levels_reward(self) -> float:
        explore_thresh = 22
        scale_factor = 4
        level_sum = self.get_levels_sum()
        if level_sum < explore_thresh:
            scaled = level_sum
        else:
            scaled = (level_sum-explore_thresh) / scale_factor + explore_thresh
        self.max_level_rew = max(self.max_level_rew,
                                 scaled,
                                 )
        return self.max_level_rew

    def get_knn_reward(self) -> float:

        pre_rew = self.explore_weight * 0.005
        post_rew = self.explore_weight * 0.01
        # self.khn_index.get_current_count returns int
        cur_size = self.knn_index.get_current_count() if self.use_screen_explore else len(self.seen_coords)
        base = (self.base_explore if self.levels_satisfied else cur_size) * pre_rew
        post = (cur_size if self.levels_satisfied else 0) * post_rew
        return base + post

    def get_badges(self):
        return self.__class__.bit_count(self.read_m(0xD356))

    def read_party(self):
        return [self.read_m(addr) for addr in [0xD164,
                                               0xD165,
                                               0xD166,
                                               0xD167,
                                               0xD168,
                                               0xD169,
                                               ]]

    def update_heal_reward(self,
                           current_health: float,
                           last_health: float,
                           before_party_size: int,
                           after_party_size: int,
                           ):
        # if health increased and party size did not change
        if (current_health > last_health and
                before_party_size == after_party_size):
            if last_health > 0:
                heal_amount = current_health - last_health
                if heal_amount > 0.5:
                    print(f'healed: {heal_amount}')
                    self.save_screenshot('healing')
                self.total_healing_rew += heal_amount * 4
            else:
                self.died_count += 1

    def get_all_events_reward(self) -> int:
        # adds up all event flags, exclude museum ticket
        event_flags_start = 0xD747
        event_flags_end = 0xD886
        museum_ticket = (0xD754, 0)
        base_event_flags = 13
        return max(
            sum(
                [
                    self.__class__.bit_count(self.read_m(i))
                    for i in range(event_flags_start, event_flags_end)
                ]
            )
            - base_event_flags
            - int(self.read_bit(museum_ticket[0], museum_ticket[1])),
        0,
    )

    def get_game_state_reward(self,
                              print_stats=False,
                              ) -> Dict[str, Union[int, float]]:
        # addresses from https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map
        # https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/event_constants.asm
        '''
        num_poke = self.read_m(0xD163)
        poke_xps = [self.read_triple(a) for a in [0xD179, 0xD1A5, 0xD1D1, 0xD1FD, 0xD229, 0xD255]]
        #money = self.read_money() - 975 # subtract starting money
        seen_poke_count = sum([self.bit_count(self.read_m(i)) for i in range(0xD30A, 0xD31D)])
        all_events_score = sum([self.bit_count(self.read_m(i)) for i in range(0xD747, 0xD886)])
        oak_parcel = self.read_bit(0xD74E, 1)
        oak_pokedex = self.read_bit(0xD74B, 5)
        opponent_level = self.read_m(0xCFF3)
        self.max_opponent_level = max(self.max_opponent_level, opponent_level)
        enemy_poke_count = self.read_m(0xD89C)
        self.max_opponent_poke = max(self.max_opponent_poke, enemy_poke_count)

        if print_stats:
            print(f'num_poke : {num_poke}')
            print(f'poke_levels : {poke_levels}')
            print(f'poke_xps : {poke_xps}')
            #print(f'money: {money}')
            print(f'seen_poke_count : {seen_poke_count}')
            print(f'oak_parcel: {oak_parcel} oak_pokedex: {oak_pokedex} all_events_score: {all_events_score}')
        '''

        state_scores = {'event': self.reward_scale*self.update_max_event_rew(),
                        #'party_xp': self.reward_scale*0.1*sum(poke_xps),
                        'level': self.reward_scale*self.get_levels_reward(),
                        'heal': self.reward_scale*self.total_healing_rew,
                        'op_lvl': self.reward_scale*self.update_max_op_level(),
                        'dead': self.reward_scale*-0.1*self.died_count,
                        'badge': self.reward_scale*self.get_badges() * 5,
                        #'op_poke': self.reward_scale*self.max_opponent_poke * 800,
                        #'money': self.reward_scale* money * 3,
                        #'seen_poke': self.reward_scale * seen_poke_count * 400,
                        'explore': self.reward_scale * self.get_knn_reward(),
        }

        return state_scores

    def save_screenshot(self,
                        name: str,
                        ) -> None:
        ss_dir = self.s_path / Path('screenshots')
        ss_dir.mkdir(exist_ok=True)
        path = ss_dir / Path(f'frame{self.instance_id}_r{self.total_reward:.4f}_{self.reset_count}_{name}.jpeg')
        plt.imsave(path,
                   self.render(reduce_res=False),
                   )

    def update_max_op_level(self) -> float:
        #opponent_level = self.read_m(0xCFE8) - 5 # base level
        opponent_level = max([self.read_m(a) for a in [0xD8C5,
                                                       0xD8F1,
                                                       0xD91D,
                                                       0xD949,
                                                       0xD975,
                                                       0xD9A1,
                                                       ]]) - 5
        #if opponent_level >= 7:
        #    self.save_screenshot('highlevelop')
        self.max_opponent_level = max(self.max_opponent_level,
                                      opponent_level,
                                      )
        return self.max_opponent_level * 0.2

    def update_max_event_rew(self) -> int:
        cur_rew = self.get_all_events_reward()
        self.max_event_rew = max(cur_rew,
                                 self.max_event_rew,
                                 )
        return self.max_event_rew

    def read_hp_fraction(self) -> float:
        hp_sum = sum([self.read_hp(add) for add in [0xD16C,
                                                    0xD198,
                                                    0xD1C4,
                                                    0xD1F0,
                                                    0xD21C,
                                                    0xD248,
                                                    ]])
        max_hp_sum = sum([self.read_hp(add) for add in [0xD18D,
                                                        0xD1B9,
                                                        0xD1E5,
                                                        0xD211,
                                                        0xD23D,
                                                        0xD269,
                                                        ]])
        max_hp_sum = max(max_hp_sum, 1)
        return hp_sum / max_hp_sum

    def read_hp(self,
                start: int,
                ) -> int:
        return 256 * self.read_m(start) + self.read_m(start+1)

    # built-in since python 3.10
    @staticmethod
    def bit_count(bits: int,
                  ) -> int:
        return bin(bits).count('1')

    def read_triple(self,
                    start_add: int,
                    ) -> int:
        return 256*256*self.read_m(start_add) + 256*self.read_m(start_add+1) + self.read_m(start_add+2)

    @staticmethod
    def read_bcd(num: int,
                 ) -> int:
        return 10 * ((num >> 4) & 0x0f) + (num & 0x0f)

    def read_money(self) -> int:
        return (100 * 100 * self.__class__.read_bcd(self.read_m(0xD347)) +
                100 * self.__class__.read_bcd(self.read_m(0xD348)) +
                self.__class__.read_bcd(self.read_m(0xD349)))

    @staticmethod
    def get_map_location(map_idx: int,
                         ) -> str:
        map_locations = {
            0: "Pallet Town",
            1: "Viridian City",
            2: "Pewter City",
            3: "Cerulean City",
            12: "Route 1",
            13: "Route 2",
            14: "Route 3",
            15: "Route 4",
            33: "Route 22",
            37: "Red house first",
            38: "Red house second",
            39: "Blues house",
            40: "oaks lab",
            41: "Pokémon Center (Viridian City)",
            42: "Poké Mart (Viridian City)",
            43: "School (Viridian City)",
            44: "House 1 (Viridian City)",
            47: "Gate (Viridian City/Pewter City) (Route 2)",
            49: "Gate (Route 2)",
            50: "Gate (Route 2/Viridian Forest) (Route 2)",
            51: "viridian forest",
            52: "Pewter Museum (floor 1)",
            53: "Pewter Museum (floor 2)",
            54: "Pokémon Gym (Pewter City)",
            55: "House with disobedient Nidoran♂ (Pewter City)",
            56: "Poké Mart (Pewter City)",
            57: "House with two Trainers (Pewter City)",
            58: "Pokémon Center (Pewter City)",
            59: "Mt. Moon (Route 3 entrance)",
            60: "Mt. Moon",
            61: "Mt. Moon",
            68: "Pokémon Center (Route 4)",
            193: "Badges check gate (Route 22)"
        }
        if map_idx in map_locations.keys():
            return map_locations[map_idx]
        else:
            return "Unknown Location"
