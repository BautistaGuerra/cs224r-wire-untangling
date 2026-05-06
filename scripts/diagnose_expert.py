"""
Diagnostic: run the expert N times and break down failures by
distance error vs orientation error vs both. No render.
"""

import argparse

import numpy as np


def main(episodes: int):
    from robosuite.wrappers import GymWrapper

    from wire_untangling.envs import StickReorderEnv
    from wire_untangling.policies import PickPlaceExpertPolicy, build_obs_index_map
    from wire_untangling.utils.transform import yaw_error_mod_pi, yaw_from_quat_wxyz

    raw = StickReorderEnv(
        robots="Panda",
        num_sticks=1,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        horizon=500,
    )
    g = GymWrapper(raw)
    obs_map = build_obs_index_map(g)
    expert = PickPlaceExpertPolicy(obs_map, goal_yaw=0.0)

    successes = 0
    fail_pos_only = 0
    fail_yaw_only = 0
    fail_both = 0
    yaw_errs_at_end = []
    dists_at_end = []

    for ep in range(episodes):
        obs, _ = g.reset()
        expert.reset()
        done = False
        while not done:
            action, _ = expert.predict(obs)
            obs, _, term, trunc, info = g.step(action)
            done = term or trunc

        body_id = raw.stick_body_ids[0]
        pos = raw.sim.data.body_xpos[body_id]
        delta = pos - raw._goal_positions[0]
        xy_dist = float(np.linalg.norm(delta[:2]))
        z_dist = float(abs(delta[2]))
        dist = float(np.linalg.norm(delta))
        yaw = yaw_from_quat_wxyz(raw.sim.data.body_xquat[body_id])
        yaw_err = yaw_error_mod_pi(yaw, raw.goal_yaw)

        dists_at_end.append(dist)
        yaw_errs_at_end.append(yaw_err)

        pos_ok = dist <= raw.success_threshold
        yaw_ok = yaw_err <= raw.orientation_threshold

        if pos_ok and yaw_ok:
            successes += 1
            tag = "OK"
        elif pos_ok:
            fail_yaw_only += 1
            tag = "FAIL yaw"
        elif yaw_ok:
            fail_pos_only += 1
            tag = "FAIL pos"
        else:
            fail_both += 1
            tag = "FAIL both"

        print(f"  ep {ep + 1:3d}: dist={dist*100:5.2f}cm "
              f"(xy={xy_dist*100:5.2f}cm z={z_dist*100:5.2f}cm) "
              f"yaw={np.rad2deg(yaw_err):5.1f}°  {tag}")

    g.close()
    print()
    print(f"Successes:           {successes}/{episodes} ({successes / episodes:.0%})")
    print(f"Fail pos only:       {fail_pos_only}")
    print(f"Fail yaw only:       {fail_yaw_only}")
    print(f"Fail both:           {fail_both}")
    print(f"Median final dist:   {np.median(dists_at_end) * 100:.2f}cm")
    print(f"Median final yaw:    {np.rad2deg(np.median(yaw_errs_at_end)):.1f}°")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=30)
    args = parser.parse_args()
    main(args.episodes)
