

from mahad.IncreasedStateBattlePolicy import ISDQNBattlePolicy

from mahad.TBPolicy import  MaxPowerMovesTeamBuildPolicy

from vgc.competition.Competitor import Competitor
from vgc.behaviour.BattlePolicies import BattlePolicy
from vgc.behaviour.TeamBuildPolicies import TeamBuildPolicy


class MahadAgent(Competitor):

    def __init__(self, name: "Mahad"):
        self.name_ = name
        self.my_battle_policy = ISDQNBattlePolicy(load_model=True,model_path='MixedApproachModel.pth')
        self.my_team_build_policy = MaxPowerMovesTeamBuildPolicy()

    @property
    def battle_policy(self) -> BattlePolicy:
        return self.my_battle_policy

    @property
    def team_build_policy(self) -> TeamBuildPolicy:
        return self.my_team_build_policy

    @property
    def name(self) -> str:
        return self.name_
