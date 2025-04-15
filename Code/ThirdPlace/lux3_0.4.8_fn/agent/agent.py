import torch

from .base import Global, log
from .state import State
from .fleet import find_hidden_constants
from .tasks import find_moves


class Agent:
    def __init__(self, player: str, env_cfg, weights_dir: str) -> None:

        self.team_id = 0 if player == "player_0" else 1
        self.opp_team_id = 1 - self.team_id

        log(f"init player {self.team_id}")
        log(f"weights_dir = {weights_dir}")
        log(f"env_cfg = {env_cfg}")

        self.unit_model = torch.jit.load(f"{weights_dir}/unit_unet.pth")
        self.unit_model.eval()

        self.sap_model = torch.jit.load(f"{weights_dir}/sap_unet.pth")
        self.sap_model.eval()

        Global.MAX_UNITS = env_cfg["max_units"]
        Global.UNIT_MOVE_COST = env_cfg["unit_move_cost"]
        Global.UNIT_SAP_COST = env_cfg["unit_sap_cost"]
        Global.UNIT_SAP_RANGE = env_cfg["unit_sap_range"]
        Global.UNIT_SENSOR_RANGE = env_cfg["unit_sensor_range"]

        self.state = State(self.team_id)
        self.previous_state = self.state.copy()

    def act(self, step: int, obs, remaining_overage_time: int = 60):
        state = self.state
        self.previous_state = state.copy()

        state.update(obs)

        log(
            f"start step {state.global_step:>3}"
            f", match {state.match_number}:{state.match_step:>3}"
            f", wins {Global.NUM_WINS}/{Global.NUM_COMPLETED_MATCHES}"
            f", points {state.fleet.points:>3}:{state.opp_fleet.points:>3}"
            f", status {state.get_game_status():>2}:{state.get_match_status():>2}"
        )

        if state.match_step == 0:
            self.previous_state = state.copy()

        find_hidden_constants(self.previous_state, state)

        # state.show_visible_map()
        # state.show_visible_energy_field()
        # state.show_explored_map()
        # state.show_explored_energy_field()
        # state.show_exploration_map()

        find_moves(self)

        # state.show_tasks(show_path=False)

        return state.create_actions_array()
