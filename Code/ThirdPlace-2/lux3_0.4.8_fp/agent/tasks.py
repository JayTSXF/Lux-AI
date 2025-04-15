from collections import defaultdict

import torch
import numpy as np
from scipy.signal import convolve2d

from .base import (
    log,
    Global,
    SPACE_SIZE,
    clip_int,
    transpose,
    get_opposite,
    nearby_positions,
    get_nebula_tile_drift_speed,
)
from .path import Action, ActionType, path_to_actions


def find_moves(agent):
    if Global.Params.IL:
        apply_nn(agent.state, agent.previous_state, agent.unit_model, agent.sap_model)
    else:
        apply_rb(agent.state)


def apply_rb(state):
    pass


def apply_nn(state, previous_state, unit_model, sap_model):
    with torch.no_grad():
        obs, gf = create_unit_nn_input(state, previous_state)
        p = unit_model(
            torch.from_numpy(obs).unsqueeze(0), torch.from_numpy(gf).unsqueeze(0)
        )
    policy = p.squeeze(0).numpy()

    node_to_ships = defaultdict(list)
    for ship in sorted(state.fleet, key=lambda s: -s.energy):
        node_to_ships[ship.node].append(ship)

    sap_ships = []
    for node, ships in node_to_ships.items():
        if len(ships) > 1:
            ships = ships[: len(ships) // 2]

        for ship in ships:
            if set_action(state, ship, policy) == ActionType.sap:
                sap_ships.append(ship)

    if sap_ships:
        apply_nn_sap_tasks(state, previous_state, sap_model, sap_ships)

    policy[ActionType.sap] = -10000
    for ship in sap_ships:
        if not ship.action_queue:
            new_action = set_action(state, ship, policy)
            log(
                f"step={state.global_step}: Can't apply sap action for {ship}, "
                f"found a new action {new_action}",
                level=2,
            )


def set_action(state, ship, policy):
    if state.team_id == 0:
        x, y = ship.coordinates
        predictions = policy[:, y, x].tolist()
        action_to_score = dict(zip(ActionType, predictions))
    else:
        x, y = get_opposite(*ship.coordinates)
        predictions = policy[:, y, x].tolist()
        action_to_score = {}
        for action, score in zip(ActionType, predictions):
            action = ActionType(action).transpose(reflective=True).value
            action_to_score[action] = score

    if not ship.can_sap():
        action_to_score.pop(ActionType.sap.value)

    best_score = max(action_to_score.values())
    best_action = [x for x, v in action_to_score.items() if v == best_score][0]

    if best_action == ActionType.sap:
        pass
    else:
        ship.action_queue = [Action(ActionType(best_action))]

    return best_action


def apply_nn_sap_tasks(state, previous_state, sap_model, sap_ships):
    sap_ships = sorted(sap_ships, key=lambda x: x.energy)

    for ship in sap_ships:
        obs, gf = create_sap_nn_input(state, previous_state, ship)
        with torch.no_grad():
            p = sap_model(
                torch.from_numpy(obs).unsqueeze(0),
                torch.from_numpy(gf).unsqueeze(0),
            )
            p = torch.sigmoid(p)
            sap_policy = p.squeeze(0).squeeze(0).numpy()
            if state.team_id == 1:
                sap_policy = transpose(sap_policy, reflective=True)

        r = Global.UNIT_SAP_RANGE * 2 + 1
        sap_kernel = np.ones((r, r), dtype=np.float32)

        ship_sap_field = np.zeros((SPACE_SIZE, SPACE_SIZE), np.float32)
        ship_sap_field[ship.node.y, ship.node.x] = 1
        ship_sap_field = convolve2d(
            ship_sap_field,
            sap_kernel,
            mode="same",
            boundary="fill",
            fillvalue=0,
        )

        sap_policy = sap_policy * ship_sap_field

        max_policy = sap_policy.max()
        # log(f"{ship} sap with prob {max_policy}")
        if max_policy > 0.5:
            sap_y, sap_x = np.where(sap_policy == max_policy)
            sap_y, sap_x = int(sap_y[0]), int(sap_x[0])
            if state.field.sap_mask[sap_y, sap_x] == 0:
                log(
                    f"Ignore a pointless sap action: {ship} -> {sap_x, sap_y}, step {state.global_step}",
                    level=1,
                )
            else:
                dx = sap_x - ship.node.x
                dy = sap_y - ship.node.y
                ship.action_queue = [Action(ActionType.sap, dx, dy)]


def get_sap_array(previous_state):
    sap_array = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.float32)
    for unit in previous_state.fleet:
        if (
            unit.action_queue
            and unit.action_queue[0].type == ActionType.sap
            and unit.node is not None
            and unit.can_sap()
        ):
            action = unit.action_queue[0]
            sap_x = unit.node.x + action.dx
            sap_y = unit.node.y + action.dy
            for x, y in nearby_positions(sap_x, sap_y, 1):
                if x == sap_x and y == sap_y:
                    sap_array[y, x] += 1
                else:
                    sap_array[y, x] += Global.UNIT_SAP_DROPOFF_FACTOR

    sap_array *= Global.UNIT_SAP_COST / Global.MAX_UNIT_ENERGY

    # if sap_array.any():
    #     show_field(sap_array * 400)

    return sap_array


def create_unit_nn_input(state, previous_state):
    r = Global.UNIT_SAP_RANGE * 2 + 1
    sap_kernel = np.ones((r, r), dtype=np.int32)

    gf = np.zeros((17, 3, 3), dtype=np.float32)

    if (
        Global.OBSTACLE_MOVEMENT_PERIOD_FOUND
        and Global.OBSTACLE_MOVEMENT_DIRECTION_FOUND
    ):
        nebula_tile_drift_direction = 1 if get_nebula_tile_drift_speed() > 0 else -1
        num_steps_before_obstacle_movement = state.num_steps_before_obstacle_movement()
    else:
        nebula_tile_drift_direction = 0
        num_steps_before_obstacle_movement = -Global.MAX_STEPS_IN_MATCH

    gf[0] = nebula_tile_drift_direction
    gf[1] = (
        Global.NEBULA_ENERGY_REDUCTION / Global.MAX_UNIT_ENERGY
        if Global.NEBULA_ENERGY_REDUCTION_FOUND
        else -1
    )
    gf[2] = Global.UNIT_MOVE_COST / Global.MAX_UNIT_ENERGY
    gf[3] = Global.UNIT_SAP_COST / Global.MAX_UNIT_ENERGY
    gf[4] = Global.UNIT_SAP_RANGE / Global.SPACE_SIZE
    gf[5] = (
        Global.UNIT_SAP_DROPOFF_FACTOR if Global.UNIT_SAP_DROPOFF_FACTOR_FOUND else -1
    )
    gf[6] = (
        Global.UNIT_ENERGY_VOID_FACTOR if Global.UNIT_ENERGY_VOID_FACTOR_FOUND else -1
    )
    gf[7] = state.match_step / Global.MAX_STEPS_IN_MATCH
    gf[8] = state.match_number / Global.NUM_MATCHES_IN_GAME
    gf[9] = num_steps_before_obstacle_movement / Global.MAX_STEPS_IN_MATCH
    gf[10] = state.fleet.points / 1000
    gf[11] = max(state.fleet.points - 5, state.opp_fleet.points) / 1000
    gf[12] = state.fleet.reward / 1000
    gf[13] = state.opp_fleet.reward / 1000
    gf[14] = sum(Global.RELIC_RESULTS) / 3
    gf[15] = min(Global.NEBULA_VISION_REDUCTION_OPTIONS) / 8
    gf[16] = (Global.ALL_RELICS_FOUND,)

    d = np.zeros((28, SPACE_SIZE, SPACE_SIZE), dtype=np.float32)

    for unit in state.fleet:
        if unit.energy >= 0:
            x, y = unit.coordinates
            d[0, y, x] += 1
            d[1, y, x] += unit.energy

    for unit in state.opp_fleet:
        if unit.energy >= 0:
            x, y = unit.coordinates
            d[2, y, x] += 1
            d[3, y, x] += unit.energy

    for unit in previous_state.fleet:
        if unit.energy >= 0:
            x, y = unit.coordinates
            d[4, y, x] += 1
            d[5, y, x] += unit.energy

    for unit in previous_state.opp_fleet:
        if unit.energy >= 0:
            x, y = unit.coordinates
            d[6, y, x] += 1
            d[7, y, x] += unit.energy

    for i in [0, 2, 4, 6]:
        d[i] /= 10

    for i in [1, 3, 5, 7]:
        d[i] /= Global.MAX_UNIT_ENERGY

    d[8] = get_sap_array(previous_state)

    f = state.field
    d[9] = f.vision
    d[10] = f.energy / Global.MAX_UNIT_ENERGY
    d[11] = f.asteroid
    d[12] = f.nebulae
    d[13] = f.relic
    d[14] = f.reward
    d[15] = f.need_to_explore_for_relic
    d[16] = f.need_to_explore_for_reward
    d[17] = f.num_units_in_sap_range / 10
    d[18] = f.num_opp_units_in_sap_range / 10
    d[19] = f.fleet_vision(state.opp_fleet, min(Global.NEBULA_VISION_REDUCTION_OPTIONS))
    d[20] = (state.global_step - f.last_relic_check) / Global.MAX_STEPS_IN_MATCH
    d[21] = (state.global_step - f.last_step_in_vision) / Global.MAX_STEPS_IN_MATCH

    # 22 - out of vision opp unit position
    # 23 - out of vision opp unit energy
    for unit in state.opp_fleet.ships:
        if (
            unit.node is not None
            and unit.energy >= 0
            and unit.steps_since_last_seen > 0
        ):
            x, y = unit.coordinates
            d[22, y, x] += 1
            d[23, y, x] += unit.energy

    d[22] /= 10
    d[23] /= Global.MAX_UNIT_ENERGY

    # 24 - unit wo energy count
    # 25 - can move
    # 26 - can sap
    # 27 - sap count
    for unit in state.fleet:
        x, y = unit.coordinates
        if unit.energy < 0:
            d[24, y, x] += 1
        if unit.energy >= Global.UNIT_MOVE_COST:
            d[25, y, x] += 1
        if unit.energy >= Global.UNIT_SAP_COST:
            d[26, y, x] += 1
            d[27, y, x] += 1

    d[27] = convolve2d(
        d[27],
        sap_kernel,
        mode="same",
        boundary="fill",
        fillvalue=0,
    )

    d[24] /= 10
    d[25] /= 10
    d[26] /= 10
    d[27] /= 10

    if state.team_id == 1:
        d = transpose(d, reflective=True).copy()

    return d, gf


def create_sap_nn_input(state, previous_state, sap_ship):

    gf = np.zeros((17, 3, 3), dtype=np.float32)

    if Global.OBSTACLE_MOVEMENT_PERIOD_FOUND:
        nebula_tile_drift_direction = 1 if get_nebula_tile_drift_speed() > 0 else -1
        num_steps_before_obstacle_movement = state.num_steps_before_obstacle_movement()
    else:
        nebula_tile_drift_direction = 0
        num_steps_before_obstacle_movement = -Global.MAX_STEPS_IN_MATCH

    gf[0] = nebula_tile_drift_direction
    gf[1] = (
        Global.NEBULA_ENERGY_REDUCTION / Global.MAX_UNIT_ENERGY
        if Global.NEBULA_ENERGY_REDUCTION_FOUND
        else -1
    )
    gf[2] = Global.UNIT_MOVE_COST / Global.MAX_UNIT_ENERGY
    gf[3] = Global.UNIT_SAP_COST / Global.MAX_UNIT_ENERGY
    gf[4] = Global.UNIT_SAP_RANGE / Global.SPACE_SIZE
    gf[5] = (
        Global.UNIT_SAP_DROPOFF_FACTOR if Global.UNIT_SAP_DROPOFF_FACTOR_FOUND else -1
    )
    gf[6] = (
        Global.UNIT_ENERGY_VOID_FACTOR if Global.UNIT_ENERGY_VOID_FACTOR_FOUND else -1
    )
    gf[7] = state.match_step / Global.MAX_STEPS_IN_MATCH
    gf[8] = state.match_number / Global.NUM_MATCHES_IN_GAME
    gf[9] = num_steps_before_obstacle_movement / Global.MAX_STEPS_IN_MATCH
    gf[10] = state.fleet.points / 1000
    gf[11] = max(state.fleet.points - 5, state.opp_fleet.points) / 1000
    gf[12] = state.fleet.reward / 1000
    gf[13] = state.opp_fleet.reward / 1000
    gf[14] = sum(Global.RELIC_RESULTS) / 3
    gf[15] = sap_ship.energy / Global.MAX_UNIT_ENERGY
    gf[16] = min(Global.NEBULA_VISION_REDUCTION_OPTIONS) / 8

    energy_field = state.field.energy
    nebulae_field = state.field.nebulae

    d = np.zeros((29, SPACE_SIZE, SPACE_SIZE), dtype=np.float32)

    r = Global.UNIT_SAP_RANGE * 2 + 1
    sap_kernel = np.ones((r, r), dtype=np.int32)
    ship_arr = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.int32)
    ship_arr[sap_ship.node.y, sap_ship.node.x] = 1
    ship_arr = convolve2d(
        ship_arr,
        sap_kernel,
        mode="same",
        boundary="fill",
        fillvalue=0,
    )

    d[0] = ship_arr

    unit_sap_dropoff_factor = (
        Global.UNIT_SAP_DROPOFF_FACTOR if Global.UNIT_SAP_DROPOFF_FACTOR_FOUND else 0.5
    )

    other_saps = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.float16)
    for unit in state.fleet:
        if unit.energy >= 0:
            x, y = unit.coordinates

            if unit.action_queue and unit.action_queue[0].type == ActionType.sap:
                action = unit.action_queue[0]
                dx, dy = action.dx, action.dy
                sap_x, sap_y = x + dx, y + dy

                for x, y in nearby_positions(sap_x, sap_y, 1):
                    if x == sap_x and y == sap_y:
                        other_saps[y, x] += 1
                    else:
                        other_saps[y, x] += unit_sap_dropoff_factor

    other_saps *= Global.UNIT_SAP_COST / Global.MAX_UNIT_ENERGY

    d[1] = other_saps

    # 2 - num units
    # 3 - unit energy
    for unit in state.fleet:
        if unit.energy >= 0:
            x, y = unit.coordinates
            d[2, y, x] += 1
            d[3, y, x] += unit.energy

    d[2] /= 10
    d[3] /= Global.MAX_UNIT_ENERGY

    # 4 - opp unit position
    # 5 - opp unit energy
    for unit in state.opp_fleet:
        if unit.energy >= 0:
            x, y = unit.coordinates
            d[4, y, x] += 1
            d[5, y, x] += unit.energy

    d[4] /= 10
    d[5] /= Global.MAX_UNIT_ENERGY

    # 6 - num units next step
    # 7 - unit energy next step
    for unit in state.fleet:
        if unit.energy >= 0:
            x, y = unit.coordinates

            if unit.action_queue:
                action_type = ActionType(unit.action_queue[0].type)
            else:
                action_type = ActionType.sap

            dx, dy = action_type.to_direction()
            next_x = clip_int(x + dx)
            next_y = clip_int(y + dy)

            next_energy = unit.energy + energy_field[next_y, next_x]
            if action_type == ActionType.sap:
                next_energy -= Global.UNIT_SAP_COST
            elif action_type != ActionType.center:
                next_energy -= Global.UNIT_MOVE_COST

            if nebulae_field[next_y, next_x]:
                next_energy -= Global.NEBULA_ENERGY_REDUCTION

            d[6, next_y, next_x] += 1
            d[7, next_y, next_x] += next_energy

    d[6] /= 10
    d[7] /= Global.MAX_UNIT_ENERGY

    # 8 - previous step unit positions
    # 9 - previous step unit energy
    for unit in previous_state.fleet:
        if unit.energy >= 0:
            x, y = unit.coordinates
            d[8, y, x] += 1
            d[9, y, x] += unit.energy

    d[8] /= 10
    d[9] /= Global.MAX_UNIT_ENERGY

    # 10 - previous step opp unit positions
    # 11 - previous step opp unit energy
    for unit in previous_state.opp_fleet:
        if unit.energy >= 0:
            x, y = unit.coordinates
            d[10, y, x] += 1
            d[11, y, x] += unit.energy

    d[10] /= 10
    d[11] /= Global.MAX_UNIT_ENERGY

    d[12] = get_sap_array(previous_state)

    f = state.field
    d[13] = f.vision
    d[14] = f.energy / Global.MAX_UNIT_ENERGY
    d[15] = f.asteroid
    d[16] = f.nebulae
    d[17] = f.relic
    d[18] = f.reward
    d[19] = f.need_to_explore_for_relic
    d[20] = f.need_to_explore_for_reward
    d[21][sap_ship.node.y, sap_ship.node.x] = 1  # unit position
    d[22] = f.num_units_in_sap_range / 10
    d[23] = f.num_opp_units_in_sap_range / 10
    d[24] = f.fleet_vision(state.opp_fleet, min(Global.NEBULA_VISION_REDUCTION_OPTIONS))
    d[25] = (state.global_step - f.last_relic_check) / Global.MAX_STEPS_IN_MATCH
    d[26] = (state.global_step - f.last_step_in_vision) / Global.MAX_STEPS_IN_MATCH

    # 27 - out of vision opp unit position
    # 28 - out of vision opp unit energy
    for unit in state.opp_fleet.ships:
        if (
            unit.node is not None
            and unit.energy >= 0
            and unit.steps_since_last_seen > 0
        ):
            x, y = unit.coordinates
            d[27, y, x] += 1
            d[28, y, x] += unit.energy

    d[27] /= 10
    d[28] /= Global.MAX_UNIT_ENERGY

    if state.team_id == 1:
        d = transpose(d, reflective=True).copy()

    return d, gf
