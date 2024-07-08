import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import namedtuple, deque

from mahad.HeuristicBattlePolicy import HeuristicBattlePolicy
from vgc.behaviour import BattlePolicy
from vgc.datatypes.Objects import GameState, Pkm


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'terminated'))
type_chart = {
    'NORMAL': {'ROCK': 0.5, 'GHOST': 0.0, 'STEEL': 0.5},
    'FIRE': {'FIRE': 0.5, 'WATER': 0.5, 'GRASS': 2.0, 'ICE': 2.0, 'BUG': 2.0, 'ROCK': 0.5, 'DRAGON': 0.5, 'STEEL': 2.0},
    'WATER': {'FIRE': 2.0, 'WATER': 0.5, 'GRASS': 0.5, 'GROUND': 2.0, 'ROCK': 2.0, 'DRAGON': 0.5},
    'ELECTRIC': {'WATER': 2.0, 'ELECTRIC': 0.5, 'GRASS': 0.5, 'GROUND': 0.0, 'FLYING': 2.0, 'DRAGON': 0.5},
    'GRASS': {'FIRE': 0.5, 'WATER': 2.0, 'GRASS': 0.5, 'POISON': 0.5, 'GROUND': 2.0, 'FLYING': 0.5, 'BUG': 0.5, 'ROCK': 2.0, 'DRAGON': 0.5, 'STEEL': 0.5},
    'ICE': {'FIRE': 0.5, 'WATER': 0.5, 'GRASS': 2.0, 'ICE': 0.5, 'GROUND': 2.0, 'FLYING': 2.0, 'DRAGON': 2.0, 'STEEL': 0.5},
    'FIGHTING': {'NORMAL': 2.0, 'ICE': 2.0, 'POISON': 0.5, 'FLYING': 0.5, 'PSYCHIC': 0.5, 'BUG': 0.5, 'ROCK': 2.0, 'GHOST': 0.0, 'DARK': 2.0, 'STEEL': 2.0, 'FAIRY': 0.5},
    'POISON': {'GRASS': 2.0, 'POISON': 0.5, 'GROUND': 0.5, 'ROCK': 0.5, 'GHOST': 0.5, 'STEEL': 0.0, 'FAIRY': 2.0},
    'GROUND': {'FIRE': 2.0, 'ELECTRIC': 2.0, 'GRASS': 0.5, 'POISON': 2.0, 'FLYING': 0.0, 'BUG': 0.5, 'ROCK': 2.0, 'STEEL': 2.0},
    'FLYING': {'ELECTRIC': 0.5, 'GRASS': 2.0, 'FIGHTING': 2.0, 'BUG': 2.0, 'ROCK': 0.5, 'STEEL': 0.5},
    'PSYCHIC': {'FIGHTING': 2.0, 'POISON': 2.0, 'PSYCHIC': 0.5, 'DARK': 0.0, 'STEEL': 0.5},
    'BUG': {'FIRE': 0.5, 'GRASS': 2.0, 'FIGHTING': 0.5, 'POISON': 0.5, 'FLYING': 0.5, 'PSYCHIC': 2.0, 'GHOST': 0.5, 'DARK': 2.0, 'STEEL': 0.5, 'FAIRY': 0.5},
    'ROCK': {'FIRE': 2.0, 'ICE': 2.0, 'FIGHTING': 0.5, 'GROUND': 0.5, 'FLYING': 2.0, 'BUG': 2.0, 'STEEL': 0.5},
    'GHOST': {'NORMAL': 0.0, 'PSYCHIC': 2.0, 'GHOST': 2.0, 'DARK': 0.5},
    'DRAGON': {'DRAGON': 2.0, 'STEEL': 0.5, 'FAIRY': 0.0},
    'DARK': {'FIGHTING': 0.5, 'PSYCHIC': 2.0, 'GHOST': 2.0, 'DARK': 0.5, 'FAIRY': 0.5},
    'STEEL': {'FIRE': 0.5, 'WATER': 0.5, 'ELECTRIC': 0.5, 'ICE': 2.0, 'ROCK': 2.0, 'STEEL': 0.5, 'FAIRY': 2.0},
    'FAIRY': {'FIRE': 0.5, 'FIGHTING': 2.0, 'POISON': 0.5, 'DRAGON': 2.0, 'DARK': 2.0, 'STEEL': 0.5}
}

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, state_dim=32, action_dim=6):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, x):
        return self.network(x)


class ISDQNBattlePolicy(BattlePolicy):
    def __init__(self, state_dim=32, action_dim=6, lr=0.01, gamma=0.99, epsilon_start=1.0, epsilon_end=0.1,
                 epsilon_decay=50000, load_model=False, model_path=None):
        super(ISDQNBattlePolicy, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        # my status
        self.previous_type = None
        self.previous_hp = None
        self.myPkm_id = None
        self.previous_status = None
        # opp prev Status
        self.opp_previous_type = None
        self.opp_previous_hp = None
        self.oppPkm_id = None
        self.opp_previous_status = None
        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayMemory(10000)
        self.steps_done = 0

        if load_model and model_path:
            self.load_model(model_path)


    def get_type_multiplier(self,attacking_type, defending_type):
        return type_chart.get(attacking_type, {}).get(defending_type, 1.0)

    def has_type_advantage(self,attacker, defender):
        multiplier = self.get_type_multiplier(attacker.type.name, defender.type.name)
        return multiplier > 1.0

    def has_type_disadvantage(self,attacker, defender):
       multiplier = self.get_type_multiplier(attacker.type.name, defender.type.name)
       return multiplier < 1.0

    def can_move(self,mypkm : Pkm):
        if mypkm.frozen() or mypkm.asleep() or mypkm.paralyzed() or mypkm.fainted():
            return 0
        else:
            return 1

    def should_switch(self, current_pkm, opponent_pkm, party_pkm):

        if self.has_type_disadvantage(current_pkm,opponent_pkm):
            if self.has_type_advantage(party_pkm[0],opponent_pkm):
                return 4
            elif self.has_type_advantage(party_pkm[1],opponent_pkm):
                return 5

        if current_pkm.hp < 0.3 * current_pkm.max_hp:
            if party_pkm[0].hp> 0.5*party_pkm[0].max_hp :
                return 4
            elif party_pkm[1].hp> 0.5*party_pkm[0].max_hp :
                return 5

        if not self.can_move(current_pkm):
            if party_pkm[0].hp> 0.5*party_pkm[0].max_hp :
                return 4
            elif party_pkm[1].hp> 0.5*party_pkm[0].max_hp :
                return 5

        return 0

    def heuristic_action(self, state:GameState):

        return self.should_switch(state.teams[0].active,state.teams[1].active,state.teams[0].party)

    def get_state(self, g: GameState):

        my_team = g.teams[0].active
        opp_team = g.teams[1].active

        # if my and opps type are same
        are_same = 1 if my_team.type == opp_team.type else 0

        if self.previous_type != my_team.type:
            self.opp_previous_hp = None
            self.previous_status = None
        if self.opp_previous_type != opp_team.type:
            self.opp_previous_hp = None
            self.opp_previous_status = None

        my_current_hp = my_team.hp
        opp_current_hp = opp_team.hp
        # creates Opponent hp change

        if self.opp_previous_hp is None:
            del_opp_hp = 0
        else:
            del_opp_hp = self.opp_previous_hp - opp_team.hp

        # my hp change
        if self.previous_hp is None:
            del_hp = 0
        else:
            del_hp = my_team.hp - self.previous_hp

        # change in status

        if my_team.asleep() or my_team.fainted() or my_team.paralyzed() or my_team.frozen() and self.previous_status == 1:
            del_status = 0
        elif my_team.asleep() or my_team.fainted() or my_team.paralyzed() or my_team.frozen() and self.previous_status == 0:
            del_status = -1
        elif not my_team.asleep() and not my_team.fainted() and not my_team.paralyzed() and not my_team.frozen() and self.previous_status == 0:
            del_status = 1
        else:
            del_status = 0

        if opp_team.asleep() or opp_team.fainted() or opp_team.paralyzed() or opp_team.frozen() and self.opp_previous_status == 1:
            del_opp_status = 0
        elif opp_team.asleep() or opp_team.fainted() or opp_team.paralyzed() or opp_team.frozen() and self.opp_previous_status == 0:
            del_opp_status = 1
        elif not opp_team.asleep() and not opp_team.fainted() and not opp_team.paralyzed() and not opp_team.frozen() and not self.opp_previous_status == 0:
            del_opp_status = -1
        else:
            del_opp_status = 0

        """------------------------------------------------------------"""
        # setting previous to current hps
        if my_team.asleep() or my_team.fainted() or my_team.paralyzed() or my_team.frozen():
            self.previous_status = 1
        else:
            self.previous_status = 0

        self.opp_previous_hp = opp_current_hp
        self.previous_hp = my_current_hp
        self.previous_type = my_team.type
        self.opp_previous_type = opp_team.type
        """"---------------------------------------------------------------"""
        active_moves = g.teams[0].active.moves
        party_1moves = g.teams[0].party[0].moves
        party_2moves = g.teams[0].party[1].moves
        features = np.array([
            g.weather.condition,
            del_hp,
            my_team.type.real,
            are_same,
            del_status,
            del_opp_hp,
            opp_team.type.real,
            del_opp_status,
            active_moves[0].type,
            active_moves[1].type,
            active_moves[2].type,
            active_moves[3].type,
            party_1moves[0].type,
            party_1moves[1].type,
            party_1moves[2].type,
            party_1moves[3].type,
            party_2moves[0].type,
            party_2moves[1].type,
            party_2moves[2].type,
            party_2moves[3].type,
            active_moves[0].power,
            active_moves[1].power,
            active_moves[2].power,
            active_moves[3].power,
            party_1moves[0].power,
            party_1moves[1].power,
            party_1moves[2].power,
            party_1moves[3].power,
            party_2moves[0].power,
            party_2moves[1].power,
            party_2moves[2].power,
            party_2moves[3].power,

             ])

        return features

    def get_action(self, g: GameState):


        current_state = self.get_state(g)
        state_tensor = torch.tensor(np.array([current_state]), dtype=torch.float)

        action = self.select_action(state_tensor,g)
        if action == 4 or action == 5 :

            heuristic = self.heuristic_action(g)
            if heuristic != 0:
               return action
            else:
               return torch.tensor([[heuristic]], dtype= torch.long)
        else:
            return action

    def select_action(self, state_tensor, g:GameState):
        sample = random.random()
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(
            -1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state_tensor).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[HeuristicBattlePolicy().get_action(g)]], dtype=torch.long)

    def calculate_reward(self, state: GameState, next_state: GameState,r) -> float:
        my_pkm = next_state.teams[0].active
        opp_pkm = next_state.teams[1].active

        my_current_hp = state.teams[0].active.hp
        my_next_hp = next_state.teams[0].active.hp
        opp_current_hp = state.teams[1].active.hp
        opp_next_hp = next_state.teams[1].active.hp

        del_hp = my_next_hp - my_current_hp
        del_opp_hp = opp_current_hp - opp_next_hp

        reward = del_opp_hp - del_hp

        # Different weights for different status conditions
        status_penalties = {
            'asleep': -5,
            'fainted': -20,
            'frozen': -15,
            'paralyzed': -5
        }

        # Apply penalty for my Pokémon's status conditions
        if my_pkm.asleep():
            reward += status_penalties['asleep']
        if my_pkm.fainted():
            reward += status_penalties['fainted']
        if my_pkm.frozen():
            reward += status_penalties['frozen']
        if my_pkm.paralyzed():
            reward += status_penalties['paralyzed']

        # Apply reward for opponent's status conditions
        if opp_pkm.asleep():
            reward -= status_penalties['asleep']
        if opp_pkm.fainted():
            reward -= status_penalties['fainted']
        if opp_pkm.frozen():
            reward -= status_penalties['frozen']
        if opp_pkm.paralyzed():
            reward -= status_penalties['paralyzed']


        return reward+r


    def optimize_model(self, batch_size):

        if len(self.memory) < batch_size:
            return
        transitions = self.memory.sample(batch_size)

        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat(
            [torch.tensor(np.array([s]), dtype=torch.float) for s in batch.next_state if s is not None])
        state_batch = torch.cat([torch.tensor(np.array([s]), dtype=torch.float) for s in batch.state])
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(batch_size)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def update_target_net(self):

        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, model_path):
        torch.save(self.policy_net.state_dict(), model_path)

    def load_model(self, model_path):
        self.policy_net.load_state_dict(torch.load(model_path))