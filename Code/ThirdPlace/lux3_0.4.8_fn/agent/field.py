import numpy as np
from functools import cached_property
from collections import defaultdict

from scipy.signal import convolve2d

from .path import NodeType
from .base import log, Global, SPACE_SIZE, Colors, nearby_positions


class StaticField:
    DISTANCE = None
    OPP_DISTANCE = None

    REWARD_DISTANCE_NUM_REWARD_NODES = 0
    REWARD_DISTANCE = None


class Field:
    def __init__(self, state, previous_field=None):

        self._state = state

        (
            self.asteroid,
            self.nebulae,
            self.energy,
            self.energy_gain,
            self.last_relic_check,
            self.last_step_in_vision,
            self.need_to_explore_for_relic,
            self.need_to_explore_for_reward,
            self.unknown_energy,
            self.unknown_nodes,
        ) = self._create_space_fields()

        self.last_time_seen = self._get_last_time_seen_field(previous_field)

    @property
    def space(self):
        return self._state.space

    def _create_space_fields(self):
        asteroid_field = create_empty_field()
        nebulae_field = create_empty_field()
        energy_field = create_empty_field()
        energy_field[:] = Global.HIDDEN_NODE_ENERGY
        energy_gain_field = create_empty_field()
        last_relic_check = create_empty_field()
        last_step_in_vision = create_empty_field()
        need_to_explore_for_relic = create_empty_field()
        need_to_explore_for_relic[:] = 1
        need_to_explore_for_reward = create_empty_field()
        need_to_explore_for_reward[:] = 1
        unknown_energy = create_empty_field()
        unknown_nodes = create_empty_field()
        for node in self.space:
            x, y = node.coordinates
            if node.type == NodeType.asteroid:
                asteroid_field[y, x] = 1
            elif node.type == NodeType.nebula:
                nebulae_field[y, x] = 1
            if node.energy is not None:
                energy_field[y, x] = node.energy
            energy_gain_field[y, x] = node.energy_gain
            last_relic_check[y, x] = node.last_relic_check
            last_step_in_vision[y, x] = node.last_step_in_vision
            if node.explored_for_relic:
                need_to_explore_for_relic[y, x] = 0
            if node.explored_for_reward:
                need_to_explore_for_reward[y, x] = 0
            if node.energy is None:
                unknown_energy[y, x] = 1
            if node.is_unknown:
                unknown_nodes[y, x] = 1

        return (
            asteroid_field,
            nebulae_field,
            energy_field,
            energy_gain_field,
            last_relic_check,
            last_step_in_vision,
            need_to_explore_for_relic,
            need_to_explore_for_reward,
            unknown_energy,
            unknown_nodes,
        )

    @cached_property
    def vision(self):
        field = create_empty_field()
        for node in self.space:
            if node.is_visible:
                field[node.y, node.x] = 1
        return field

    @cached_property
    def sensor_power(self):
        return self._sensor_power(self._state.fleet)

    @cached_property
    def opp_sensor_power(self):
        return self._sensor_power(self._state.opp_fleet)

    @staticmethod
    def _sensor_power(fleet):
        r = Global.UNIT_SENSOR_RANGE * 2 + 1
        sensor_kernel = np.zeros((r, r), dtype=np.float32)
        for d in range(0, Global.UNIT_SENSOR_RANGE + 1):
            sensor_kernel[d : r - d, d : r - d] = d + 1

        field = create_empty_field()
        for unit in fleet:
            x, y = unit.coordinates
            field[y, x] += 1

        field = convolve2d(
            field,
            sensor_kernel,
            mode="same",
            boundary="fill",
            fillvalue=0,
        )

        for unit in fleet:
            x, y = unit.coordinates
            field[y, x] += 10

        return field

    def fleet_vision(self, fleet, nebula_vision_reduction):
        sensor_power = self._sensor_power(fleet)

        vision = create_empty_field()
        for node in self.space:
            x, y = node.coordinates
            sp = sensor_power[y, x]
            if sp < 1:
                continue

            reduction = 0 if node.type != NodeType.nebula else nebula_vision_reduction
            is_visible = sp - reduction >= 1
            if is_visible:
                vision[y, x] = 1

        return vision

    @cached_property
    def opp_vision(self):
        field = create_empty_field()
        field[np.where(self._state.opp_fleet.vision > 0)] = 1

        num_opp_ships_with_rewards, opp_reward_nodes = prob_opp_on_rewards(self._state)
        if num_opp_ships_with_rewards and opp_reward_nodes:
            prob = num_opp_ships_with_rewards / len(opp_reward_nodes)
            if prob == 1:
                for node in opp_reward_nodes:
                    for x, y in nearby_positions(
                        *node.coordinates, Global.UNIT_SENSOR_RANGE
                    ):
                        field[y, x] = 1
            else:
                node_to_num_probs = defaultdict(int)
                for node in opp_reward_nodes:
                    for x, y in nearby_positions(
                        *node.coordinates, Global.UNIT_SENSOR_RANGE
                    ):
                        node_to_num_probs[(x, y)] += 1

                num_rewards_without_ships = (
                    len(opp_reward_nodes) - num_opp_ships_with_rewards
                )
                for (x, y), num_probs in node_to_num_probs.items():
                    if num_probs > num_rewards_without_ships:
                        field[y, x] = 1
                    else:
                        no_vision_prob = (1 - prob) ** num_probs
                        field[y, x] = max(field[y, x], 1 - no_vision_prob)

        return field

    @property
    def distance(self):
        if StaticField.DISTANCE is None:
            field = create_empty_field()

            for x in range(SPACE_SIZE):
                for y in range(SPACE_SIZE):
                    field[y, x] = self._state.fleet.spawn_distance(x, y)

            StaticField.DISTANCE = field

        return StaticField.DISTANCE

    @cached_property
    def opp_distance(self):
        if StaticField.OPP_DISTANCE is None:
            field = create_empty_field()

            for x in range(SPACE_SIZE):
                for y in range(SPACE_SIZE):
                    field[y, x] = self._state.opp_fleet.spawn_distance(x, y)

            StaticField.OPP_DISTANCE = field

        return StaticField.OPP_DISTANCE

    @cached_property
    def num_units_in_sap_range(self):
        return self._num_units_in_sap_range(self._state.fleet)

    @cached_property
    def num_opp_units_in_sap_range(self):
        return self._num_units_in_sap_range(self._state.opp_fleet)

    @staticmethod
    def _num_units_in_sap_range(fleet):
        field = create_empty_field()
        for unit in fleet:
            x, y = unit.node.coordinates
            field[y, x] += 1

        sap_kernel = np.ones((3, 3), dtype=np.float32)
        field = convolve2d(
            field,
            sap_kernel,
            mode="same",
            boundary="fill",
            fillvalue=0,
        )
        return field

    @property
    def control(self):
        return self.vision

    @cached_property
    def opp_ships_potential_positions(self):
        out_of_control = np.logical_not(self.control)

        # is it possible for opponent's ships to reach this position
        field = np.logical_and(
            out_of_control, self.opp_distance <= self._state.match_step
        )

        # add opponent's ships with energy
        for ship in self._state.opp_fleet:
            if ship.energy >= 0:
                x, y = ship.coordinates
                field[y, x] = 1

        return field

    @cached_property
    def opp_sap_ships_potential_positions(self):
        out_of_control = np.logical_not(self.control)

        # is it possible for opponent's ships to reach this position
        field = np.logical_and(
            out_of_control, self.opp_distance <= self._state.match_step
        )

        # add opponent's ships that can sap
        for ship in self._state.opp_fleet:
            if ship.can_sap():
                x, y = ship.coordinates
                field[y, x] = 1

        return field

    @cached_property
    def sap_mask(self):
        # returns positions that make sense to sap
        sap_kernel = np.ones((5, 5), dtype=np.float32)

        field = convolve2d(
            self.opp_ships_potential_positions,
            sap_kernel,
            mode="same",
            boundary="fill",
            fillvalue=0,
        )
        field = field > 0
        return field

    @cached_property
    def reward(self):
        field = create_empty_field()
        for node in self.space.reward_nodes:
            field[node.y, node.x] = 1
        return field

    @cached_property
    def relic(self):
        field = create_empty_field()
        for node in self.space.relic_nodes:
            field[node.y, node.x] = 1
        return field

    def _get_last_time_seen_field(self, previous_field):
        if previous_field is None or self._state.match_step == 0:
            field = create_empty_field()
            field[:] = Global.MAX_STEPS_IN_MATCH
            return field

        field = previous_field.last_time_seen

        field += 1
        field *= np.logical_not(self.vision)
        field = np.clip(field, a_min=0, a_max=Global.MAX_STEPS_IN_MATCH)

        return field


def create_empty_field():
    return np.zeros((SPACE_SIZE, SPACE_SIZE), np.float32)


def show_field(weights):
    def add_color(i):
        color = Colors.green if i > 0 else Colors.red
        return f"{color}{i:>3}{Colors.endc}"

    line = " + " + " ".join([f"{x:>2}" for x in range(Global.SPACE_SIZE)]) + "  +\n"
    str_grid = line
    for y in range(Global.SPACE_SIZE):

        str_row = []

        for x in range(Global.SPACE_SIZE):
            v = int(weights[y, x])
            str_row.append(add_color(v))

        str_grid += "".join([f"{y:>2}", *str_row, f" {y:>2}", "\n"])

    str_grid += line
    log(str_grid)


def prob_opp_on_rewards(state):
    if not Global.ALL_REWARDS_FOUND:
        return 0, set()

    reward_nodes = set()
    for reward_node in state.space.reward_nodes:
        if not reward_node.is_visible:
            reward_nodes.add(reward_node)

    opp_rewards_in_vision = 0
    for ship in state.opp_fleet:
        if ship.node.reward:
            opp_rewards_in_vision = +1

    num_opp_ships_with_rewards = state.opp_fleet.reward - opp_rewards_in_vision

    return num_opp_ships_with_rewards, reward_nodes
