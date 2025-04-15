import numpy as np
from collections import defaultdict

from .base import (
    log,
    Global,
    SPACE_SIZE,
    get_spawn_location,
    nearby_positions,
    obstacles_moving,
    cardinal_positions,
    manhattan_distance,
    chebyshev_distance,
)
from .path import Action, ActionType, apply_action, actions_to_path
from .space import Node, Space, NodeType


class Fleet:
    def __init__(self, team_id):
        self.team_id: int = team_id
        self.points: int = 0
        self.reward: int = 0
        self.spawn_position = get_spawn_location(self.team_id)

        self.ships = [Ship(unit_id) for unit_id in range(Global.MAX_UNITS)]

        self.vision = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.int8)

    def __repr__(self):
        return f"Fleet({self.team_id})"

    def __iter__(self):
        for ship in self.ships:
            if ship.is_visible:
                yield ship

    def update(self, obs, space: Space):
        points = int(obs["team_points"][self.team_id])
        self.reward = max(0, points - self.points)
        self.points = points

        for ship, visible, position, energy in zip(
            self.ships,
            obs["units_mask"][self.team_id],
            obs["units"]["position"][self.team_id],
            obs["units"]["energy"][self.team_id],
        ):
            if visible:
                ship.node = space.get_node(*position)
                ship.energy = int(energy)
                ship.steps_since_last_seen = 0
            else:
                if (
                    ship.node is not None
                    and ship.energy >= 0
                    and not space.get_node(*ship.coordinates).is_visible
                ):
                    # The ship is out of sight of our sensors
                    ship.steps_since_last_seen += 1
                    ship.task = None
                    ship.action_queue = []
                else:
                    # The ship was probably destroyed.
                    ship.clear()

            ship.action_queue = []
            ship.update_vision()

        self.update_vision()

    def clear(self):
        self.points = 0
        self.reward = 0
        for ship in self.ships:
            ship.clear()

    def expected_sensor_mask(self):
        space_size = Global.SPACE_SIZE
        sensor_range = Global.UNIT_SENSOR_RANGE
        mask = np.zeros((space_size, space_size), dtype=np.int16)
        for ship in self:
            x, y = ship.coordinates
            for _y in range(
                max(0, y - sensor_range), min(space_size, y + sensor_range + 1)
            ):
                mask[_y][
                    max(0, x - sensor_range) : min(space_size, x + sensor_range + 1)
                ] = 1
        return mask

    def update_vision(self):
        self.vision[:] = 0

        for ship in self.ships:
            self.vision += ship.vision

    def spawn_distance(self, x, y):
        return manhattan_distance(self.spawn_position, (x, y))


class Ship:
    def __init__(self, unit_id: int):
        self.unit_id = unit_id
        self.energy = 0
        self.node: Node | None = None
        self.steps_since_last_seen: int = 0

        self.task = None
        self.action_queue: list[Action] = []

        self.vision = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.int8)

    def __repr__(self):
        position = self.node.coordinates if self.node else None
        return f"Ship({self.unit_id}, node={position}, energy={self.energy})"

    @property
    def is_visible(self) -> True:
        return self.node is not None and self.steps_since_last_seen == 0

    @property
    def coordinates(self):
        return self.node.coordinates if self.node else None

    def path(self):
        if not self.action_queue:
            return [self.coordinates]
        return actions_to_path(self.coordinates, self.action_queue)

    def clear(self):
        self.energy = 0
        self.node = None
        self.task = None
        self.action_queue = []
        self.steps_since_last_seen = 0

    def can_move(self) -> bool:
        return self.node is not None and self.energy >= Global.UNIT_MOVE_COST

    def can_sap(self) -> bool:
        return self.node is not None and self.energy >= Global.UNIT_SAP_COST

    def next_position(self) -> tuple[int, int]:
        if not self.can_move() or not self.action_queue:
            return self.coordinates
        return apply_action(*self.coordinates, action=self.action_queue[0].type)

    def update_vision(self):
        self.vision[:] = 0

        if not self.is_visible:
            return

        x, y = self.coordinates
        r = Global.UNIT_SENSOR_RANGE

        self.vision[
            max(0, y - r) : min(SPACE_SIZE, y + r + 1),
            max(0, x - r) : min(SPACE_SIZE, x + r + 1),
        ] = 1


def find_hidden_constants(previous_state, state):
    """
    Attempts to discover hidden constants by observing interactions
    between ships and nebulae (NEBULA_ENERGY_REDUCTION) and
    between ships and opponent's ships (UNIT_SAP_DROPOFF_FACTOR, UNIT_ENERGY_VOID_FACTOR).
    """
    _find_nebula_energy_reduction(previous_state, state)
    find_nebula_vision_reduction(state)
    _find_ship_interaction_constants(previous_state, state)


def find_nebula_vision_reduction(state):
    if Global.NEBULA_VISION_REDUCTION_FOUND:
        return

    if not Global.NEBULA_VISION_REDUCTION_OPTIONS:
        return

    if state.can_obstacles_move_this_step():
        return

    sensor_power = state.field.sensor_power
    for node in state.space:
        sp = sensor_power[node.y, node.x]
        if sp <= 0:
            continue

        suitable_options = []
        if node.is_visible and node.type == NodeType.nebula:
            for option in Global.NEBULA_VISION_REDUCTION_OPTIONS:
                if sp - option >= 1:
                    suitable_options.append(option)
        elif not node.is_visible:
            for option in Global.NEBULA_VISION_REDUCTION_OPTIONS:
                if sp - option < 1:
                    suitable_options.append(option)
        else:
            continue

        obs = f"node = {node.coordinates}, sensor power = {sp}, visible = {node.is_visible}"

        if not suitable_options:
            log(
                f"Can't find an nebula vision reduction, which would fits to the observation: {obs}",
                level=1,
            )

        if len(suitable_options) == 1:
            v = suitable_options[0]
            log(
                f"There is only one nebula tile vision reduction ({v}), "
                f"that fit the observation: {obs}",
            )
            log(f"Find param NEBULA_VISION_REDUCTION = {v}")
            Global.NEBULA_VISION_REDUCTION = v
            Global.NEBULA_VISION_REDUCTION_FOUND = True
            Global.NEBULA_VISION_REDUCTION_OPTIONS = [v]
            return

        if len(suitable_options) < len(Global.NEBULA_VISION_REDUCTION_OPTIONS):
            log(
                f"There are {len(suitable_options)} obstacle movement periods ({suitable_options}), "
                f"that fit the observation: {obs}"
            )
            Global.NEBULA_VISION_REDUCTION_OPTIONS = suitable_options


def _find_nebula_energy_reduction(previous_state, state):
    if Global.NEBULA_ENERGY_REDUCTION_FOUND:
        return

    spawn_position = state.fleet.spawn_position
    opp_void_positions = set()
    for ship in state.opp_fleet:
        if ship.energy > 0:
            for x, y in cardinal_positions(*ship.coordinates):
                opp_void_positions.add((x, y))

    check_previous_type = False
    if not Global.OBSTACLE_MOVEMENT_PERIOD_FOUND:
        check_previous_type = True
    if Global.OBSTACLE_MOVEMENT_PERIOD_FOUND and obstacles_moving(state.global_step):
        check_previous_type = True

    for previous_ship, ship in zip(previous_state.fleet.ships, state.fleet.ships):
        if not previous_ship.is_visible or not ship.is_visible:
            continue

        node = ship.node
        if node.energy is None:
            continue

        if ship.coordinates == spawn_position or ship.coordinates in opp_void_positions:
            continue

        move_cost = 0
        if node != previous_ship.node:
            move_cost = Global.UNIT_MOVE_COST

        if previous_ship.energy < 30 - move_cost:
            continue

        if node.type != NodeType.nebula:
            continue

        if (
            check_previous_type
            and previous_state.space.get_node(*node.coordinates).type != NodeType.nebula
        ):
            continue

        delta = previous_ship.energy - ship.energy + node.energy - move_cost

        if delta == 25:
            Global.NEBULA_ENERGY_REDUCTION = 25
        elif 0 <= delta <= 5:
            Global.NEBULA_ENERGY_REDUCTION = delta
        else:
            log(
                f"Can't find NEBULA_ENERGY_REDUCTION with ship = {ship}, "
                f"delta = {delta}, step = {state.global_step}",
                level=1,
            )
            continue

        Global.NEBULA_ENERGY_REDUCTION_FOUND = True

        log(
            f"Find param NEBULA_ENERGY_REDUCTION = {Global.NEBULA_ENERGY_REDUCTION}",
            level=2,
        )
        return


def _find_ship_interaction_constants(previous_state, state):
    if Global.UNIT_SAP_DROPOFF_FACTOR_FOUND and Global.UNIT_ENERGY_VOID_FACTOR_FOUND:
        return

    sap_coordinates = []
    for previous_ship in previous_state.fleet:
        action = None
        if previous_ship.action_queue:
            action = previous_ship.action_queue[0]

        if (
            action is None
            or action.type != ActionType.sap
            or not previous_ship.can_sap()
        ):
            continue

        x, y = previous_ship.coordinates
        dx, dy = action.dx, action.dy

        sap_coordinates.append((x + dx, y + dy))

    void_field = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.int16)
    for previous_ship, ship in zip(previous_state.fleet.ships, state.fleet.ships):
        if not previous_ship.is_visible or previous_ship.energy <= 0:
            continue

        node = ship.node
        move_cost = 0
        if node is None:
            next_position = previous_ship.next_position()
            node = state.space.get_node(*next_position)
        elif node != previous_ship.node:
            move_cost = Global.UNIT_MOVE_COST

        for x_, y_ in cardinal_positions(*node.coordinates):
            void_field[x_, y_] += previous_ship.energy - move_cost

    direct_sap_hits = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.int16)
    adjacent_sap_hits = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.int16)
    for x, y in sap_coordinates:
        for x_, y_ in nearby_positions(x, y, distance=1):
            if x_ == x and y_ == y:
                direct_sap_hits[x_, y_] += 1
            else:
                adjacent_sap_hits[x_, y_] += 1

    additional_energy_loss = find_additional_energy_loss(previous_state, state)

    if not Global.UNIT_SAP_DROPOFF_FACTOR_FOUND:
        _find_unit_sap_dropoff_factor(
            previous_state,
            state,
            void_field,
            direct_sap_hits,
            adjacent_sap_hits,
            additional_energy_loss,
        )

    if not Global.UNIT_ENERGY_VOID_FACTOR_FOUND:
        _find_unit_energy_void_factor(
            previous_state,
            state,
            void_field,
            direct_sap_hits,
            adjacent_sap_hits,
            additional_energy_loss,
        )


def _can_opp_sap(previous_opp_ship, opp_ship, additional_energy_loss):
    if not previous_opp_ship.can_sap():
        return False

    if opp_ship.node != previous_opp_ship.node:
        return False

    for my_ship, energy_loss in additional_energy_loss.items():
        if (
            chebyshev_distance(my_ship.coordinates, opp_ship.coordinates)
            <= Global.UNIT_SAP_RANGE + 1
        ):
            return True

    return False


def _find_unit_sap_dropoff_factor(
    previous_state,
    state,
    void_field,
    direct_sap_hits,
    adjacent_sap_hits,
    additional_energy_loss,
):
    for previous_opp_ship, opp_ship in zip(
        previous_state.opp_fleet.ships, state.opp_fleet.ships
    ):
        if not previous_opp_ship.is_visible or not opp_ship.is_visible:
            continue

        if opp_ship.energy <= 0:
            continue

        x, y = opp_ship.coordinates
        if (
            void_field[x, y] == 0
            and adjacent_sap_hits[x, y] > 0
            and not _can_opp_sap(previous_opp_ship, opp_ship, additional_energy_loss)
        ):
            num_direct_hits = int(direct_sap_hits[x, y])
            num_adjacent_hits = int(adjacent_sap_hits[x, y])

            node = opp_ship.node
            if node.energy is None:
                continue

            move_cost = 0
            if node != previous_opp_ship.node:
                move_cost = Global.UNIT_MOVE_COST

            if previous_state.space.get_node(*node.coordinates).type == NodeType.nebula:
                if not Global.NEBULA_ENERGY_REDUCTION_FOUND:
                    continue
                move_cost += Global.NEBULA_ENERGY_REDUCTION

            delta = (
                previous_opp_ship.energy
                - opp_ship.energy
                + node.energy
                - move_cost
                - num_direct_hits * Global.UNIT_SAP_COST
            )

            delta_per_hit = delta / num_adjacent_hits

            sap_cost = Global.UNIT_SAP_COST

            if abs(delta_per_hit - sap_cost) <= 1:
                Global.UNIT_SAP_DROPOFF_FACTOR = 1
            elif abs(delta_per_hit - sap_cost * 0.5) <= 1:
                Global.UNIT_SAP_DROPOFF_FACTOR = 0.5
            elif abs(delta_per_hit - sap_cost * 0.25) <= 1:
                Global.UNIT_SAP_DROPOFF_FACTOR = 0.25
            else:
                log(
                    f"Can't find UNIT_SAP_DROPOFF_FACTOR with ship = {opp_ship}, "
                    f"delta = {delta}, num_direct_hits = {num_direct_hits}, "
                    f"num_adjacent_hits = {num_adjacent_hits}, step = {state.global_step}",
                    level=1,
                )
                continue

            Global.UNIT_SAP_DROPOFF_FACTOR_FOUND = True

            log(
                f"Find param UNIT_SAP_DROPOFF_FACTOR = {Global.UNIT_SAP_DROPOFF_FACTOR}",
                level=2,
            )
            return


def _find_unit_energy_void_factor(
    previous_state,
    state,
    void_field,
    direct_sap_hits,
    adjacent_sap_hits,
    additional_energy_loss,
):
    position_to_unit_count = defaultdict(int)
    for opp_ship in state.opp_fleet:
        position_to_unit_count[opp_ship.coordinates] += 1

    for previous_opp_ship, opp_ship in zip(
        previous_state.opp_fleet.ships, state.opp_fleet.ships
    ):
        if not previous_opp_ship.is_visible or not opp_ship.is_visible:
            continue

        if opp_ship.energy <= 0:
            continue

        x, y = opp_ship.coordinates
        if void_field[x, y] > 0 and not _can_opp_sap(
            previous_opp_ship, opp_ship, additional_energy_loss
        ):
            num_direct_hits = int(direct_sap_hits[x, y])
            num_adjacent_hits = int(adjacent_sap_hits[x, y])

            if num_adjacent_hits > 0 and not Global.UNIT_SAP_DROPOFF_FACTOR_FOUND:
                continue

            hit_loss = int(
                Global.UNIT_SAP_COST
                * (num_direct_hits + num_adjacent_hits * Global.UNIT_SAP_DROPOFF_FACTOR)
            )

            node_void_field = int(void_field[x, y])
            node_unit_count = position_to_unit_count[(x, y)]

            node = opp_ship.node
            if node.energy is None:
                continue

            move_cost = 0
            if node != previous_opp_ship.node:
                move_cost = Global.UNIT_MOVE_COST

            if previous_state.space.get_node(*node.coordinates).type == NodeType.nebula:
                if not Global.NEBULA_ENERGY_REDUCTION_FOUND:
                    continue
                move_cost += Global.NEBULA_ENERGY_REDUCTION

            delta = (
                previous_opp_ship.energy
                - opp_ship.energy
                + node.energy
                - move_cost
                - hit_loss
            )

            options = [0.0625, 0.125, 0.25, 0.375]
            results = []
            for option in options:
                expected = node_void_field / node_unit_count * option
                result = abs(expected - delta) <= 1
                results.append(result)

            if sum(results) == 1:
                for option, result in zip(options, results):
                    if result:
                        Global.UNIT_ENERGY_VOID_FACTOR = option

                Global.UNIT_ENERGY_VOID_FACTOR_FOUND = True

                log(
                    f"Find param UNIT_ENERGY_VOID_FACTOR = {Global.UNIT_ENERGY_VOID_FACTOR}",
                    level=2,
                )
                return

            if sum(results) == 0:
                log(
                    f"Can't find UNIT_ENERGY_VOID_FACTOR with ship = {opp_ship}, "
                    f"delta = {delta}, void_field = {node_void_field}, "
                    f"num_direct_hits = {num_direct_hits}, num_adjacent_hits = {num_adjacent_hits}, "
                    f"step = {state.global_step}",
                    level=1,
                )


def find_additional_energy_loss(previous_state, state):
    ship_to_energy_loss = {}

    for previous_ship, ship in zip(previous_state.fleet.ships, state.fleet.ships):
        if not previous_ship.is_visible or not ship.is_visible:
            continue

        node = ship.node

        if node.energy is None:
            continue

        if previous_ship.energy <= 0:
            continue

        move_cost = 0
        if node != previous_ship.node:
            move_cost = Global.UNIT_MOVE_COST
        elif (
            previous_ship.action_queue
            and previous_ship.can_sap()
            and previous_ship.action_queue[0].type == ActionType.sap
        ):
            move_cost = Global.UNIT_SAP_COST

        if previous_state.space.get_node_type(*node.coordinates) == NodeType.nebula:
            if not Global.NEBULA_ENERGY_REDUCTION_FOUND:
                continue
            move_cost += Global.NEBULA_ENERGY_REDUCTION

        delta = previous_ship.energy - ship.energy + node.energy - move_cost

        if delta > 0:
            ship_to_energy_loss[ship] = delta

    return ship_to_energy_loss
