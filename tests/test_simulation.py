import numpy as np

from simulation.car_simulator import CarSimulator
from simulation.obstacle_avoidance import compute_goal_vector, obstacle_avoidance_control, run_step
from simulation.visualize import scale_boxes


def test_goal_vector_is_unit():
    vec = compute_goal_vector((0, 0), (10, 0))
    assert np.allclose(vec, [1.0, 0.0], atol=1e-6)


def test_control_returns_unit_vector():
    sal = np.zeros((100, 100), np.float32)
    steer = obstacle_avoidance_control((10, 10), (90, 90), sal)
    assert abs(np.linalg.norm(steer) - 1.0) < 1e-3


def test_run_step_stays_in_bounds():
    sal = np.zeros((50, 50), np.float32)
    pos, _ = run_step((49, 49), (100, 100), sal, step_size=10)
    assert 0 <= pos[0] <= 49 and 0 <= pos[1] <= 49


def test_detection_repulsion_pushes_away():
    sal = np.zeros((200, 200), np.float32)
    # obstacle directly between car and goal, close to the car
    det = [[95, 95, 125, 125, 0.9, 1]]
    steer_clear = obstacle_avoidance_control((80, 110), (190, 110), sal)
    steer_block = obstacle_avoidance_control((80, 110), (190, 110), sal, det)
    assert not np.allclose(steer_clear, steer_block)


def test_car_reaches_goal_on_empty_map():
    sim = CarSimulator(start=(50, 400), goal=(500, 100))
    sal = np.zeros((sim.H, sim.W), np.float32)
    for _ in range(20):
        sim.update(sal, [])
        if sim.goal_reached():
            break
    assert sim.goal_reached(), f"car stuck at {sim.pos}"


def test_state_snapshot_is_copied():
    sim = CarSimulator()
    sal = np.zeros((sim.H, sim.W), np.float32)
    sim.update(sal, [], steps=3)
    s1 = sim.state()
    sim.update(sal, [], steps=3)
    s2 = sim.state()
    assert len(s2["path"]) > len(s1["path"])  # s1 must not alias the live path


def test_scale_boxes():
    boxes = [[10, 10, 20, 20, 0.9, 1]]
    scaled = scale_boxes(boxes, (100, 100), (200, 50))
    assert scaled == [[20, 5, 40, 10, 0.9, 1]]
