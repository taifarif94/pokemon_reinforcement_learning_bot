
import numpy as np
from vgc.behaviour import BattlePolicy
from vgc.datatypes.Constants import DEFAULT_PKM_N_MOVES, DEFAULT_PARTY_SIZE, TYPE_CHART_MULTIPLIER
from vgc.datatypes.Objects import GameState
from vgc.datatypes.Types import PkmStat, PkmType, WeatherCondition, PkmEntryHazard


# Utility functions
def calculate_damage(move_type, pkm_type, move_power, opp_pkm_type, attack_stage, defense_stage, weather):
    stab = 1.5 if move_type == pkm_type else 1.0
    weather_modifier = 1.0
    if (move_type == PkmType.WATER and weather == WeatherCondition.RAIN) or (
            move_type == PkmType.FIRE and weather == WeatherCondition.SUNNY):
        weather_modifier = 1.5
    elif (move_type == PkmType.WATER and weather == WeatherCondition.SUNNY) or (
            move_type == PkmType.FIRE and weather == WeatherCondition.RAIN):
        weather_modifier = 0.5

    stage_multiplier = (2 + attack_stage - defense_stage) / 2 if attack_stage >= defense_stage else 2 / (
                2 + defense_stage - attack_stage)
    type_multiplier = TYPE_CHART_MULTIPLIER[move_type][opp_pkm_type]
    damage = move_power * stab * weather_modifier * stage_multiplier * type_multiplier
    return damage


def evaluate_matchup(pkm_type, opp_pkm_type, opp_moves_types):
    max_multiplier = max(TYPE_CHART_MULTIPLIER[mtype][pkm_type] for mtype in opp_moves_types)
    return max_multiplier


class HeuristicBattlePolicy(BattlePolicy):
    def __init__(self, switch_probability=0.2, n_moves=DEFAULT_PKM_N_MOVES, n_switches=DEFAULT_PARTY_SIZE):
        super().__init__()
        self.switch_probability = switch_probability
        self.n_moves = n_moves
        self.n_switches = n_switches
        self.weather_used = {WeatherCondition.SANDSTORM: False, WeatherCondition.HAIL: False}


    def get_action(self, g: GameState) -> int:
        weather = g.weather.condition

        my_team = g.teams[0]
        my_active = my_team.active
        my_party = my_team.party
        my_attack_stage = my_team.stage[PkmStat.ATTACK]
        my_defense_stage = my_team.stage[PkmStat.DEFENSE]

        opp_team = g.teams[1]
        opp_active = opp_team.active
        opp_attack_stage = opp_team.stage[PkmStat.ATTACK]
        opp_defense_stage = opp_team.stage[PkmStat.DEFENSE]

        damage = [
            calculate_damage(move.type, my_active.type, move.power, opp_active.type, my_attack_stage, opp_defense_stage,
                             weather) for move in my_active.moves]
        best_move_id = int(np.argmax(damage))

        if damage[best_move_id] >= opp_active.hp:
            return best_move_id

        if TYPE_CHART_MULTIPLIER[my_active.moves[best_move_id].type][opp_active.type] == 2.0:
            return best_move_id

        defensive_matchup = evaluate_matchup(my_active.type, opp_active.type, [move.type for move in opp_active.moves])

        if defensive_matchup <= 1.0:
            if opp_team.entry_hazard != PkmEntryHazard.SPIKES and len(
                    opp_team.get_not_fainted()) > DEFAULT_PARTY_SIZE / 2:
                for i, move in enumerate(my_active.moves):
                    if move.hazard == PkmEntryHazard.SPIKES:
                        return i

            if not self.weather_used[
                WeatherCondition.SANDSTORM] and weather != WeatherCondition.SANDSTORM and defensive_matchup < 1.0:
                sandstorm_move = next(
                    (i for i, move in enumerate(my_active.moves) if move.weather == WeatherCondition.SANDSTORM), -1)
                if sandstorm_move != -1 and sum(1 for pkm in my_party if pkm.type in [PkmType.GROUND, PkmType.STEEL,
                                                                                      PkmType.ROCK] and not pkm.fainted()) > 2:
                    self.weather_used[WeatherCondition.SANDSTORM] = True
                    return sandstorm_move

            if not self.weather_used[
                WeatherCondition.HAIL] and weather != WeatherCondition.HAIL and defensive_matchup < 1.0:
                hail_move = next((i for i, move in enumerate(my_active.moves) if move.weather == WeatherCondition.HAIL),
                                 -1)
                if hail_move != -1 and sum(1 for pkm in my_party if pkm.type == PkmType.ICE and not pkm.fainted()) > 2:
                    self.weather_used[WeatherCondition.HAIL] = True
                    return hail_move

            if opp_attack_stage == 0 and opp_defense_stage == 0:
                debuff_move = next((i for i, move in enumerate(my_active.moves) if
                                    move.target == 1 and move.stage != 0 and move.stat in [PkmStat.ATTACK,
                                                                                           PkmStat.DEFENSE]), -1)
                if debuff_move != -1:
                    return debuff_move

            return best_move_id

        # best_switch = min((i for i, pkm in enumerate(my_party) if not pkm.fainted()),
        #                   key=lambda i: evaluate_matchup(my_party[i].type, opp_active.type,
        #                                                  [move.type for move in opp_active.moves]))
        #
        # if my_party[best_switch] != my_active:
        #     return best_switch + 4

        return best_move_id


