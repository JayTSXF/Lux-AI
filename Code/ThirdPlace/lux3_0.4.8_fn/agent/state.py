from copy import deepcopy

import numpy as np
from collections import defaultdict

from .base import (
    log,
    Global,
    Colors,
    set_game_prams,
    elements_moving,
    get_spawn_location,
    chebyshev_distance,
    obstacles_moving,
)
from .path import ActionType
from .space import Space, NodeType
from .fleet import Fleet
from .field import Field


class State:
    def __init__(self, team_id):
        self.team_id = team_id
        self.opp_team_id = 1 - team_id
        self.global_step = 0  # global step of the game
        self.match_step = 0  # current step in the match
        self.match_number = 0  # current match number

        self.space = Space()
        self.fleet = Fleet(team_id)
        self.opp_fleet = Fleet(self.opp_team_id)

        self.grid = None
        self.field = None

    def update(self, obs):

        if obs["steps"] > 0:
            self._update_step_counters()

        assert obs["steps"] == self.global_step
        assert obs["match_steps"] == self.match_step

        self._update_info_about_completed_matches(obs)

        if self.match_step == 0:
            self.fleet.clear()
            self.opp_fleet.clear()
            self.space.clear(self.global_step)
            self.space.clear_exploration_info(self.global_step)
            self.space.move_obstacles(self.global_step)
            self._update_game_params()
            self.field = Field(self)
            if self.match_number > 0 and not Global.OBSTACLE_MOVEMENT_PERIOD_FOUND:
                Global.OBSTACLES_MOVEMENT_STATUS.append(None)
            return

        points = int(obs["team_points"][self.team_id])
        reward = max(0, points - self.fleet.points)

        opp_points = int(obs["team_points"][1 - self.team_id])
        opp_reward = max(0, opp_points - self.opp_fleet.points)

        self.space.update(
            self.global_step,
            obs,
            team_id=self.team_id,
            team_reward=reward,
            opp_team_id=1 - self.team_id,
            opp_team_reward=opp_reward,
        )
        self.fleet.update(obs, self.space)
        self.opp_fleet.update(obs, self.space)

        if not self.can_obstacles_move_this_step():
            self.space.update_nodes_by_expected_sensor_mask(
                self.fleet.expected_sensor_mask()
            )

        self._update_game_params()

        self.field = Field(self, self.field)

    def _update_game_params(self):

        if self.is_game_over():
            set_game_prams(Global.GameOverParams)
            return

        if self.match_step == 0:
            set_game_prams(Global.DefaultParams)
            return

    def _update_step_counters(self):
        self.global_step += 1
        self.match_step += 1
        if self.match_step > Global.MAX_STEPS_IN_MATCH:
            self.match_step = 0
            self.match_number += 1

    def _update_info_about_completed_matches(self, obs):
        if self.match_step == Global.MAX_STEPS_IN_MATCH:
            Global.POINTS.append(self.fleet.points)
            Global.OPP_POINTS.append(self.opp_fleet.points)
        if self.match_step == 0 and self.match_number >= 1:
            team_wins = obs["team_wins"]
            Global.NUM_COMPLETED_MATCHES = sum(team_wins)
            Global.NUM_WINS = int(team_wins[self.team_id])

    def steps_left_in_match(self) -> int:
        return Global.MAX_STEPS_IN_MATCH - self.match_step

    def get_fleet(self, team_id):
        assert team_id == 0 or team_id == 1
        return self.fleet if team_id == self.team_id else self.opp_fleet

    def copy(self) -> "State":
        copy_state = State(self.team_id)

        copy_state.global_step = 0
        copy_state.match_step = 0
        copy_state.match_number = 0

        copy_state.space = deepcopy(self.space)
        copy_state.fleet = deepcopy(self.fleet)
        copy_state.opp_fleet = deepcopy(self.opp_fleet)

        return copy_state

    def num_steps_before_obstacle_movement(self):
        if not Global.OBSTACLE_MOVEMENT_PERIOD_FOUND:
            return

        d = 1
        while not elements_moving(
            self.global_step + d, Global.OBSTACLE_MOVEMENT_PERIOD
        ):
            d += 1
        return d

    def num_steps_before_energy_node_movement(self):
        if not Global.ENERGY_NODE_MOVEMENT_PERIOD_FOUND:
            return

        d = 1
        while not elements_moving(
            self.global_step + d - 1, Global.ENERGY_NODE_MOVEMENT_PERIOD
        ):
            d += 1
        return d

    def create_actions_array(self):
        ships = self.fleet.ships
        actions = np.zeros((len(ships), 3), dtype=int)

        spawn_pointer = self._get_spawn_pointer()
        if spawn_pointer:
            actions[:] = (ActionType.sap, *spawn_pointer)

        for i, ship in enumerate(ships):
            if ship.node is not None:
                actions[i][0] = ActionType.center

            if ship.action_queue:
                a = ship.action_queue[0]
                actions[i] = (a.type.value, a.dx, a.dy)

        return actions

    def _get_spawn_pointer(self):
        spawn_location = get_spawn_location(self.team_id)
        target = None
        min_distance = 0
        for ship in self.fleet:
            next_position = ship.next_position()

            if next_position == spawn_location:
                continue

            distance = chebyshev_distance(spawn_location, next_position)
            if distance > Global.UNIT_SAP_RANGE:
                continue

            if target is None or min_distance > distance:
                min_distance = distance
                target = next_position

        if target:
            return target[0] - spawn_location[0], target[1] - spawn_location[1]

    def show_visible_map(self):
        log("Visible map:")
        show_map(self.space, self.fleet, self.opp_fleet)

    def show_visible_energy_field(self):
        log("Visible energy field:")
        show_energy_field(self.space)

    def show_explored_map(self):
        log("Explored map:")
        show_map(self.space, self.fleet, self.opp_fleet, only_visible=False)

    def show_explored_energy_field(self):
        log("Explored energy field:")
        show_energy_field(self.space, only_visible=False)

    def show_exploration_map(self):
        log("Exploration map:")
        show_exploration_map(self.space)

    def show_tasks(self, show_path=False):
        log("Tasks:")
        for ship in self.fleet:
            m = f" - {ship} : {ship.task}"
            if ship.action_queue:
                a = ship.action_queue[0]
                m += f", action={a}"
            if show_path:
                m += f", {ship.path()}"
            log(m)

    def get_match_status(self):
        # returns:
        # 1 - if our agent wins
        # -1 - if the opponent wins
        # 0 - unknown

        num_relics_th = (
            2 * min(self.match_number, Global.LAST_MATCH_WHEN_RELIC_CAN_APPEAR) + 1
        )
        if len(self.space.relic_nodes) < num_relics_th:
            return 0

        if not Global.ALL_REWARDS_FOUND:
            return 0

        our_score = self.fleet.points
        opp_score = self.opp_fleet.points
        max_points_per_turn = len(self.space.reward_nodes)
        num_turns_left = self.steps_left_in_match()
        max_points_to_gain = max_points_per_turn * num_turns_left

        if our_score > opp_score + max_points_to_gain:
            return 1

        if opp_score > our_score + max_points_to_gain:
            return -1

        return 0

    def is_match_over(self):
        return self.get_match_status() != 0

    def get_game_status(self):
        # returns:
        # 1 - if our agent wins
        # -1 - if the opponent wins
        # 0 - unknown

        match_status = self.get_match_status()

        num_wins = Global.NUM_WINS
        num_losses = Global.NUM_COMPLETED_MATCHES - num_wins

        if match_status == 1:
            num_wins += 1
        elif match_status == -1:
            num_losses += 1

        num_wins_to_win = int(Global.NUM_MATCHES_IN_GAME / 2) + 1

        if num_wins >= num_wins_to_win:
            return 1

        if num_losses >= num_wins_to_win:
            return -1

        return 0

    def is_game_over(self):
        return self.get_game_status() != 0

    def can_obstacles_move_this_step(self) -> bool:
        if Global.OBSTACLE_MOVEMENT_PERIOD_FOUND:
            if obstacles_moving(self.global_step):
                return True
            return False

        if not Global.OBSTACLE_MOVEMENT_PERIOD_OPTIONS:
            return True

        if any(
            elements_moving(self.global_step, p)
            for p in Global.OBSTACLE_MOVEMENT_PERIOD_OPTIONS
        ):
            return True

        return False


def show_map(space, my_fleet=None, opp_fleet=None, only_visible=True):
    ship_signs = (
        [" "] + [str(x) for x in range(1, 10)] + ["A", "B", "C", "D", "E", "F", "H"]
    )

    my_ships = defaultdict(int)
    if my_fleet:
        for ship in my_fleet:
            my_ships[ship.node.coordinates] += 1

    opp_ships = defaultdict(int)
    if opp_fleet:
        for ship in opp_fleet:
            opp_ships[ship.node.coordinates] += 1

    line = " + " + " ".join([f"{x:>2}" for x in range(Global.SPACE_SIZE)]) + "  +\n"
    str_grid = line
    for y in range(Global.SPACE_SIZE):

        str_row = []

        for x in range(Global.SPACE_SIZE):
            node = space.get_node(x, y)

            if node.type == NodeType.unknown or (only_visible and not node.is_visible):
                str_row.append("..")
                continue

            if node.type == NodeType.nebula:
                s1 = "ñ" if node.relic else "n"
            elif node.type == NodeType.asteroid:
                s1 = "ã" if node.relic else "a"
            else:
                s1 = "~" if node.relic else " "

            if node.reward:
                if s1 == " ":
                    s1 = "_"
                s1 = f"{Colors.yellow}{s1}{Colors.endc}"

            if node.coordinates in my_ships:
                num_ships = my_ships[node.coordinates]
                s2 = f"{Colors.blue}{ship_signs[num_ships]}{Colors.endc}"
            elif node.coordinates in opp_ships:
                num_ships = opp_ships[node.coordinates]
                s2 = f"{Colors.red}{ship_signs[num_ships]}{Colors.endc}"
            else:
                s2 = " "

            str_row.append(s1 + s2)

        str_grid += " ".join([f"{y:>2}", *str_row, f"{y:>2}", "\n"])

    str_grid += line
    log(str_grid)


def show_energy_field(space, only_visible=True):
    def add_color(i):
        color = Colors.green if i > 0 else Colors.red
        return f"{color}{i:>3}{Colors.endc}"

    line = " + " + " ".join([f"{x:>2}" for x in range(Global.SPACE_SIZE)]) + "  +\n"
    str_grid = line
    for y in range(Global.SPACE_SIZE):

        str_row = []

        for x in range(Global.SPACE_SIZE):
            node = space.get_node(x, y)
            if node.energy is None or (only_visible and not node.is_visible):
                str_row.append(" ..")
            else:
                str_row.append(add_color(node.energy))

        str_grid += "".join([f"{y:>2}", *str_row, f" {y:>2}", "\n"])

    str_grid += line
    log(str_grid)


def show_exploration_map(space):
    log(
        f"all relics found: {Global.ALL_RELICS_FOUND}, "
        f"all rewards found: {Global.ALL_REWARDS_FOUND}"
    )

    line = " + " + " ".join([f"{x:>2}" for x in range(Global.SPACE_SIZE)]) + "  +\n"
    str_grid = line
    for y in range(Global.SPACE_SIZE):

        str_row = []

        for x in range(Global.SPACE_SIZE):
            node = space.get_node(x, y)
            if not node.explored_for_relic:
                s1 = "."
            else:
                s1 = "R" if node.relic else " "

            if not node.explored_for_reward:
                s2 = "."
            else:
                s2 = "P" if node.reward else " "

            str_row.append(s1 + s2)

        str_grid += " ".join([f"{y:>2}", *str_row, f"{y:>2}", "\n"])

    str_grid += line
    log(str_grid)
