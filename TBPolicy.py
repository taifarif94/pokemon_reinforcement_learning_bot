from vgc.balance.meta import MetaData
from vgc.behaviour import TeamBuildPolicy
from vgc.datatypes.Constants import DEFAULT_TEAM_SIZE
from vgc.datatypes.Objects import PkmFullTeam, PkmRoster, PkmTemplate

class MaxPowerMovesTeamBuildPolicy(TeamBuildPolicy):
    def __init__(self):
        self.roster = None

    def set_roster(self, roster: PkmRoster, ver: int = 0):
        self.roster = roster

    def get_action(self, s: MetaData) -> PkmFullTeam:
        roster = list(self.roster)
        team = heuristic_team_selection(roster)
        return team

def get_unique_move_types(moves):
    return len(set(move.type for move in moves))

def evaluate_pokemon(pokemon:PkmTemplate):
    score = 0
    score += pokemon.max_hp

    for move in pokemon.moves:
        score += move.power

    return score

def heuristic_team_selection(roster: PkmRoster) -> PkmFullTeam:

    evaluated_pokemon = [(pokemon, evaluate_pokemon(pokemon)) for pokemon in roster]

    evaluated_pokemon.sort(key=lambda x: x[1], reverse=True)

    selected_team = []

    for PkmTemplate,score in evaluated_pokemon:

         if len(selected_team) >= DEFAULT_TEAM_SIZE:
             break
         selected_team.append(PkmTemplate.gen_pkm([0,1,2,3]))

    team = PkmFullTeam(selected_team)
    return team
