import numpy as np
from copy import deepcopy
from enum import IntEnum
from scipy.signal import convolve2d

from .base import (
    log,
    Global,
    SPACE_SIZE,
    get_opposite,
    warp_point,
    nearby_positions,
    get_match_number,
    get_match_step,
    elements_moving,
    can_relic_appear,
    obstacles_moving,
    chebyshev_distance,
)


class NodeType(IntEnum):
    unknown = -1
    empty = 0
    nebula = 1
    asteroid = 2

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.type = NodeType.unknown
        self.energy = None
        self.is_visible = False
        self.last_relic_check = 0
        self.last_step_in_vision = 0

        self._relic = False
        self._reward = False
        self._explored_for_relic = False
        self._explored_for_reward = True

    def __repr__(self):
        return f"Node({self.x}, {self.y}, {self.type})"

    def __hash__(self):
        return self.coordinates.__hash__()

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    @property
    def relic(self):
        return self._relic

    @property
    def reward(self):
        return self._reward

    @property
    def explored_for_relic(self):
        return self._explored_for_relic

    @property
    def explored_for_reward(self):
        return self._explored_for_reward

    def update_relic_status(self, status: None | bool):
        if self._explored_for_relic and self._relic and not status:
            raise ValueError(
                f"Can't change the relic status {self._relic}->{status} for {self}"
                ", the tile has already been explored"
            )

        if status is None:
            self._explored_for_relic = False
            return

        self._relic = status
        self._explored_for_relic = True

    def update_reward_status(self, status: None | bool):
        if self._explored_for_reward and self._reward and not status:
            raise ValueError(
                f"Can't change the reward status {self._reward}->{status} for {self}"
                ", the tile has already been explored"
            )

        if status is None:
            self._explored_for_reward = False
            return

        self._reward = status
        self._explored_for_reward = True

    @property
    def is_unknown(self) -> bool:
        return self.type == NodeType.unknown

    @property
    def is_walkable(self) -> bool:
        return self.type != NodeType.asteroid

    @property
    def coordinates(self) -> tuple[int, int]:
        return self.x, self.y

    def manhattan_distance(self, other: "Node") -> int:
        return abs(self.x - other.x) + abs(self.y - other.y)

    @property
    def energy_gain(self):
        energy = self.energy
        if energy is None:
            energy = Global.HIDDEN_NODE_ENERGY

        if self.type == NodeType.nebula:
            energy -= Global.NEBULA_ENERGY_REDUCTION

        return energy


class Space:
    def __init__(self):
        self._nodes: list[list[Node]] = []
        for y in range(SPACE_SIZE):
            row = [Node(x, y) for x in range(SPACE_SIZE)]
            self._nodes.append(row)

        self._relic_nodes: set[Node] = set()
        self._reward_nodes: set[Node] = set()

    def __repr__(self) -> str:
        return f"Space({SPACE_SIZE}x{SPACE_SIZE})"

    def __iter__(self):
        for row in self._nodes:
            yield from row

    def get_node(self, x, y) -> Node:
        return self._nodes[y][x]

    def get_node_energy(self, x, y) -> int:
        return self._nodes[y][x].energy

    def get_node_type(self, x, y) -> NodeType:
        return self._nodes[y][x].type

    def get_opposite_node(self, x, y) -> Node:
        return self.get_node(*get_opposite(x, y))

    def update(
        self,
        global_step,
        obs,
        team_id=0,
        team_reward=0,
        opp_team_id=1,
        opp_team_reward=0,
    ):
        self.move_obstacles(global_step)
        self._update_map(global_step, obs)
        self._update_relic_map(
            global_step, obs, team_id, team_reward, opp_team_id, opp_team_reward
        )

    def _update_map(self, global_step, obs):
        sensor_mask = obs["sensor_mask"]
        obs_energy = obs["map_features"]["energy"]
        obs_tile_type = obs["map_features"]["tile_type"]

        obstacles_shifted = False
        energy_nodes_shifted = False
        for node in self:
            x, y = node.coordinates
            is_visible = sensor_mask[x, y]
            if not is_visible:
                continue

            if not node.is_unknown and node.type.value != obs_tile_type[x, y]:
                obstacles_shifted = True

            if node.energy is not None and node.energy != obs_energy[x, y]:
                energy_nodes_shifted = True

        if not Global.OBSTACLE_MOVEMENT_PERIOD_FOUND:
            self.add_obs_to_obstacles_movement_status_log(obs, obstacles_shifted)

            period = _get_obstacle_movement_period(Global.OBSTACLES_MOVEMENT_STATUS)
            if period is not None:
                Global.OBSTACLE_MOVEMENT_PERIOD_FOUND = True
                Global.OBSTACLE_MOVEMENT_PERIOD = period
                log(
                    f"Find param OBSTACLE_MOVEMENT_PERIOD = {period}",
                    level=2,
                )

        if obstacles_shifted:

            direction = _get_obstacle_movement_direction(self, obs)
            if direction:
                Global.OBSTACLE_MOVEMENT_DIRECTION_FOUND = True
                Global.OBSTACLE_MOVEMENT_DIRECTION = direction
                log(
                    f"Find param OBSTACLE_MOVEMENT_DIRECTION = {direction}",
                    level=2,
                )
                log(f"move obstacles, direction={Global.OBSTACLE_MOVEMENT_DIRECTION}")
                self.move(*Global.OBSTACLE_MOVEMENT_DIRECTION, inplace=True)
            else:
                log("Can't find OBSTACLE_MOVEMENT_DIRECTION", level=1)
                for node in self:
                    node.type = NodeType.unknown

        for node in self:
            x, y = node.coordinates
            is_visible = bool(sensor_mask[x, y])

            node.is_visible = is_visible
            if is_visible:
                node.last_relic_check = global_step
                self.get_opposite_node(x, y).last_relic_check = global_step
                node.last_step_in_vision = global_step

            if is_visible and node.is_unknown:
                node.type = NodeType(int(obs_tile_type[x, y]))

                # we can also update the node type on the other side of the map
                # because the map is symmetrical
                self.get_opposite_node(x, y).type = node.type

            if is_visible:
                node.energy = int(obs_energy[x, y])

                # the energy field should be symmetrical
                self.get_opposite_node(x, y).energy = node.energy

        if energy_nodes_shifted:
            # The energy field has changed
            # I cannot predict what the new energy field will be like.
            for node in self:
                opp_node = self.get_opposite_node(*node.coordinates)
                if not node.is_visible and not opp_node.is_visible:
                    node.energy = None
                    opp_node.energy = None

    def _update_relic_map(
        self, global_step, obs, team_id, team_reward, opp_team_id, opp_team_reward
    ):
        match = get_match_number(global_step)
        match_step = get_match_step(global_step)
        match_number = get_match_number(global_step)

        for relic_id, (mask, xy) in enumerate(
            zip(obs["relic_nodes_mask"], obs["relic_nodes"])
        ):
            if mask and not self.get_node(*xy).relic:
                # We have found a new relic.
                Global.RELIC_RESULTS[match] += 1
                self._update_relic_status(*xy, status=True)

                # We need to find reward nodes next to the relic.
                for x, y in nearby_positions(*xy, Global.RELIC_REWARD_RANGE):
                    if not self.get_node(x, y).reward:
                        self._update_reward_status(x, y, status=None)

                for reward_result in Global.REWARD_RESULTS:
                    reward_result["trust"] = False

        all_rewards_found = True

        possible_relic_spawn = can_relic_appear(global_step)

        for node in self:
            if node.is_visible and not node.explored_for_relic:
                self._update_relic_status(*node.coordinates, status=False)

            if possible_relic_spawn:
                opp_node = self.get_opposite_node(*node.coordinates)
                if not node.is_visible and not opp_node.is_visible:
                    if not node.relic and node.explored_for_relic:
                        self._update_relic_status(*node.coordinates, status=None)

            if not node.explored_for_reward:
                all_rewards_found = False

        Global.ALL_REWARDS_FOUND = all_rewards_found

        num_relics_found = sum(Global.RELIC_RESULTS)
        # the maximum number of relics (without duplicates) we can find at this stage
        num_relics_th = min(match, Global.LAST_MATCH_WHEN_RELIC_CAN_APPEAR) + 1

        if not Global.ALL_RELICS_FOUND:
            if num_relics_found >= Global.LAST_MATCH_WHEN_RELIC_CAN_APPEAR + 1:
                log(f"Found all relics: {Global.RELIC_RESULTS}", level=2)
                Global.ALL_RELICS_FOUND = True

            if match_number >= 2:
                if num_relics_found >= 1 and all(
                    node.explored_for_relic for node in self
                ):
                    log(f"Found all relics: {Global.RELIC_RESULTS}", level=2)
                    Global.ALL_RELICS_FOUND = True

        if num_relics_found >= num_relics_th or Global.ALL_RELICS_FOUND:
            for node in self:
                if not node.explored_for_relic:
                    self._update_relic_status(*node.coordinates, status=False)

        if not Global.ALL_REWARDS_FOUND:
            self._update_reward_results(obs, team_id, team_reward, full_visibility=True)
            self._filter_reward_results(global_step)
            self._update_reward_status_from_reward_results()

    def add_obs_to_obstacles_movement_status_log(self, obs, obstacles_shifted):
        if obstacles_shifted:
            Global.OBSTACLES_MOVEMENT_STATUS.append(True)
            return

        con_detect_obstacles_movements = False

        sensor_mask = obs["sensor_mask"]
        obstacles = [NodeType.nebula, NodeType.asteroid]

        for node in self:
            x, y = node.coordinates
            if (
                not sensor_mask[x, y]
                or not node.is_visible
                or node.type not in obstacles
            ):
                continue

            for dx, dy in [(1, -1), (-1, 1)]:
                x_, y_ = warp_point(x + dx, y + dy)
                next_node = self.get_node(x_, y_)
                if (
                    not sensor_mask[x_, y_]
                    or not next_node.is_visible
                    or next_node.type == node.type
                ):
                    continue

                con_detect_obstacles_movements = True
                # log(f"can detect obstacles movements with nodes {node} {next_node}")
                break

            if con_detect_obstacles_movements:
                break

        if con_detect_obstacles_movements:
            Global.OBSTACLES_MOVEMENT_STATUS.append(False)
        else:
            Global.OBSTACLES_MOVEMENT_STATUS.append(None)

    def move_obstacles(self, global_step):
        if (
            Global.OBSTACLE_MOVEMENT_PERIOD_FOUND
            and Global.OBSTACLE_MOVEMENT_DIRECTION_FOUND
            and obstacles_moving(global_step)
        ):
            log(f"move obstacles, direction={Global.OBSTACLE_MOVEMENT_DIRECTION}")
            self.move(*Global.OBSTACLE_MOVEMENT_DIRECTION, inplace=True)

    def _update_reward_status_from_relics_distribution(self):
        # Rewards can only occur near relics.
        # Therefore, if there are no relics near the node
        # we can infer that the node does not contain a reward.

        relic_map = np.zeros((SPACE_SIZE, SPACE_SIZE), np.int32)
        for node in self:
            if node.relic or not node.explored_for_relic:
                relic_map[node.y][node.x] = 1

        reward_size = 2 * Global.RELIC_REWARD_RANGE + 1

        reward_map = convolve2d(
            relic_map,
            np.ones((reward_size, reward_size), dtype=np.int32),
            mode="same",
            boundary="fill",
            fillvalue=0,
        )

        for node in self:
            if reward_map[node.y][node.x] == 0:
                # no relics in range RELIC_REWARD_RANGE
                node.update_reward_status(False)

    def _update_reward_results(self, obs, team_id, team_reward, full_visibility=True):
        ship_nodes = set()
        for active, energy, position in zip(
            obs["units_mask"][team_id],
            obs["units"]["energy"][team_id],
            obs["units"]["position"][team_id],
        ):
            if active and energy >= 0:
                # Only units with non-negative energy can give points
                ship_nodes.add(self.get_node(*position))

        if ship_nodes:
            record = {
                "step": obs["steps"],
                "nodes": ship_nodes,
                "reward": team_reward,
                "full_visibility": full_visibility,
                "trust": None,
                "known_relics": set(self.relic_nodes),
            }

            Global.REWARD_RESULTS.append(record)

    def _update_reward_status_from_reward_results(self):

        count = 0

        reward_results = list(Global.REWARD_RESULTS)

        while True:
            count += 1

            updated = False

            filtered_results = []
            for result in reward_results:

                if not result["trust"]:
                    continue

                unknown_nodes = set()
                known_reward = 0
                for n in result["nodes"]:
                    if n.explored_for_reward and not n.reward:
                        continue

                    if n.reward:
                        known_reward += 1
                        continue

                    unknown_nodes.add(n)

                if not unknown_nodes:
                    # all nodes already explored, nothing to do here
                    continue

                reward = result["reward"] - known_reward

                if reward == 0:
                    # all nodes are empty
                    for node in unknown_nodes:
                        updated = True
                        self._update_reward_status(*node.coordinates, status=False)
                    continue

                if result["full_visibility"]:
                    if reward == len(unknown_nodes):
                        # all nodes yield points
                        for node in unknown_nodes:
                            updated = True
                            self._update_reward_status(*node.coordinates, status=True)
                        continue

                    if reward > len(unknown_nodes):
                        # we shouldn't be here
                        log(
                            f"Something wrong with reward result: {result}"
                            ", this result will be ignored.",
                            level=1,
                        )
                        continue

                r = {
                    "nodes": unknown_nodes,
                    "reward": reward,
                    "full_visibility": result["full_visibility"],
                    "trust": result["trust"],
                }
                if "oov_nodes" in result:
                    r["oov_nodes"] = result["oov_nodes"]

                filtered_results.append(r)

            reward_results = filtered_results

            if not updated:
                break

            if count > 10:
                log(
                    "It takes too many cycles to _update_reward_status_from_reward_results. "
                    "It is possible that there is a bug somewhere",
                    level=1,
                )
                break

    def _update_relic_status(self, x, y, status):
        node = self.get_node(x, y)
        node.update_relic_status(status)

        opp_node = self.get_opposite_node(x, y)
        opp_node.update_relic_status(status)

        if status:
            self._relic_nodes.add(node)
            self._relic_nodes.add(opp_node)

    def _update_reward_status(self, x, y, status):
        node = self.get_node(x, y)
        node.update_reward_status(status)

        opp_node = self.get_opposite_node(x, y)
        opp_node.update_reward_status(status)

        if status:
            self._reward_nodes.add(node)
            self._reward_nodes.add(opp_node)

    @property
    def num_relics_found(self) -> int:
        return len(self._relic_nodes)

    @property
    def relic_nodes(self) -> set[Node]:
        return self._relic_nodes

    @property
    def reward_nodes(self) -> set[Node]:
        return self._reward_nodes

    def clear(self, global_step):
        for node in self:
            node.is_visible = False
            node.last_step_in_vision = global_step

    def update_nodes_by_expected_sensor_mask(self, expected_sensor_mask):
        for y in range(SPACE_SIZE):
            for x in range(SPACE_SIZE):
                if expected_sensor_mask[y][x] == 1 and self.get_node(x, y).is_unknown:
                    # Only nebulae can block vision.
                    self.get_node(x, y).type = NodeType.nebula

                    # Nebulae are symmetrical
                    self.get_opposite_node(x, y).type = NodeType.nebula

    def is_walkable(self, x, y):
        return self.get_node(x, y).is_walkable

    def move(self, dx: int, dy: int, *, inplace=False) -> "Space":
        if not inplace:
            new_space = deepcopy(self)
            for node in self:
                x, y = warp_point(node.x + dx, node.y + dy)
                new_space.get_node(x, y).type = node.type
            return new_space
        else:
            types = [n.type for n in self]
            for node, node_type in zip(self, types):
                x, y = warp_point(node.x + dx, node.y + dy)
                self.get_node(x, y).type = node_type
            return self

    def clear_exploration_info(self, global_step):
        Global.REWARD_RESULTS = []
        Global.ALL_REWARDS_FOUND = False

        if not Global.ALL_RELICS_FOUND:
            match_number = get_match_number(global_step)
            if match_number <= Global.LAST_MATCH_WHEN_RELIC_CAN_APPEAR:
                for node in self:
                    if not node.relic:
                        self._update_relic_status(node.x, node.y, status=None)

    def _filter_reward_results(self, step):

        for reward_result in Global.REWARD_RESULTS:
            if reward_result["trust"] is not None:
                continue

            result_step = reward_result["step"]
            result_match = get_match_number(result_step)

            relics_found = set()
            for relic_node in reward_result["known_relics"]:
                p = relic_node.coordinates
                if p not in relics_found and get_opposite(*p) not in relics_found:
                    relics_found.add(p)

            num_relics_found = len(relics_found)
            num_relics_th = (
                min(result_match, Global.LAST_MATCH_WHEN_RELIC_CAN_APPEAR) + 1
            )
            found_all_relics = num_relics_found >= num_relics_th

            if found_all_relics:
                nodes = []
                for node in reward_result["nodes"]:
                    if any(
                        chebyshev_distance(node.coordinates, relic_node.coordinates)
                        <= Global.RELIC_REWARD_RANGE
                        for relic_node in reward_result["known_relics"]
                    ):
                        nodes.append(node)

                reward_result["nodes"] = nodes
                reward_result["trust"] = True
            else:

                new_relics = set()
                for relic_node in self.relic_nodes:
                    if relic_node not in reward_result["known_relics"]:
                        new_relics.add(relic_node)

                node_info = []
                for node in reward_result["nodes"]:

                    within_relic_range = False
                    unknown_neighbors = False
                    within_new_relic_range = False
                    for x, y in nearby_positions(
                        *node.coordinates, Global.RELIC_REWARD_RANGE
                    ):
                        nearby_node = self.get_node(x, y)
                        if nearby_node in reward_result["known_relics"]:
                            within_relic_range = True

                        if nearby_node.last_relic_check < result_step:
                            unknown_neighbors = True

                        if nearby_node in new_relics:
                            within_new_relic_range = True

                    node_info.append(
                        {
                            "node": node,
                            "within_relic_range": within_relic_range,
                            "unknown_neighbors": unknown_neighbors,
                            "within_new_relic_range": within_new_relic_range,
                        }
                    )

                # print(f"node info (step={reward_result['step']}):", file=stderr)
                # for x in node_info:
                #     print(f" - {x}", file=stderr)

                if reward_result["reward"] == 0:
                    reward_result["nodes"] = [
                        x["node"] for x in node_info if x["within_relic_range"]
                    ]
                    reward_result["trust"] = True

                elif any(x["within_new_relic_range"] for x in node_info):
                    reward_result["trust"] = False

                elif any(x["unknown_neighbors"] for x in node_info):
                    reward_result["trust"] = None

                else:
                    reward_result["nodes"] = [
                        x["node"] for x in node_info if x["within_relic_range"]
                    ]
                    reward_result["trust"] = True


def _get_obstacle_movement_direction(space, obs):
    sensor_mask = obs["sensor_mask"]
    obs_tile_type = obs["map_features"]["tile_type"]

    suitable_directions = []
    for direction in [(1, -1), (-1, 1)]:
        moved_space = space.move(*direction, inplace=False)

        match = True
        for node in moved_space:
            x, y = node.coordinates
            if (
                sensor_mask[x, y]
                and not node.is_unknown
                and obs_tile_type[x, y] != node.type.value
            ):
                match = False
                break

        if match:
            suitable_directions.append(direction)

    if len(suitable_directions) == 1:
        return suitable_directions[0]


def _get_obstacle_movement_period(obstacles_movement_status):

    if len(obstacles_movement_status) < 5:
        return

    suitable_periods = []
    for period in Global.OBSTACLE_MOVEMENT_PERIOD_OPTIONS:

        moving_pattern = [
            elements_moving(x + 1, period)
            for x in range(len(obstacles_movement_status))
        ]
        moving_pattern[0] = False

        is_suitable = True
        obs_num_movements = 0
        pattern_num_movements = 0
        for pattern_flag, obs_flag in zip(moving_pattern, obstacles_movement_status):
            if pattern_flag is True and obs_flag is False:
                is_suitable = False
                break

            if obs_flag:
                obs_num_movements += 1
            if pattern_flag:
                pattern_num_movements += 1

        if obs_num_movements > pattern_num_movements:
            is_suitable = False

        if is_suitable:
            suitable_periods.append(period)

    Global.OBSTACLE_MOVEMENT_PERIOD_OPTIONS = suitable_periods

    if not suitable_periods:
        log(
            f"Can't find an obstacle movement period, "
            f"which would fits to the observation {obstacles_movement_status}",
            level=1,
        )
        return

    if len(suitable_periods) == 1:
        log(
            f"There is only one obstacle movement period ({suitable_periods[0]}), "
            f"that fit the observation: {obstacles_movement_status}"
        )
        return suitable_periods[0]
    else:
        log(
            f"There are {len(suitable_periods)} obstacle movement periods ({suitable_periods}), "
            f"that fit the observation: {obstacles_movement_status}"
        )
