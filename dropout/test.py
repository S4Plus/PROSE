import numpy as np
YAW_LIST = [
    -np.pi / 4,
    -np.pi / 8,
    0.0,
    np.pi / 8,
    np.pi / 4,
]

def get_pose_by_index(idx: int):
    pick_x_range = (1.062, 1.262)
    pick_y_range = (2.62, 3.12)
    place_x_range = (-0.6, -0.58)
    place_y_range = (-0.3, 0)

    cols = 4
    rows = 10

    dx = (pick_x_range[1] - pick_x_range[0]) / cols
    dy = (pick_y_range[1] - pick_y_range[0]) / rows

    # 生成 40 个单元格中心 + 28 个边界点，共 68 个测试点
    centers = [
        (pick_x_range[0] + (i + 0.5) * dx, pick_y_range[0] + (j + 0.5) * dy)
        for j in range(rows)
        for i in range(cols)
    ]
    x_grid = [pick_x_range[0] + i * dx for i in range(cols + 1)]
    y_grid = [pick_y_range[0] + j * dy for j in range(rows + 1)]
    boundary = []
    for x in x_grid:
        boundary.append((x, pick_y_range[0]))
        boundary.append((x, pick_y_range[1]))
    for y in y_grid[1:-1]:
        boundary.append((pick_x_range[0], y))
        boundary.append((pick_x_range[1], y))

    test_points = centers + boundary  # len = 68
    assert 0 <= idx < len(test_points), f"idx must be in [0, {len(test_points)})"

    px, py = test_points[idx]

    place_x = 0.5 * (place_x_range[0] + place_x_range[1])
    place_y = 0.5 * (place_y_range[0] + place_y_range[1])

    yaw = YAW_LIST[idx % len(YAW_LIST)]

    pick_quat = yaw
    place_quat = yaw

    pos_pick = np.array([px, py, 0.9])
    pos_place = np.array([place_x, place_y, 0.10493])

    pick_pose = {"position": pos_pick, "orientation": pick_quat}
    place_pose = {"position": pos_place, "orientation": place_quat}
    return pick_pose, place_pose



for i in range(68):
    pose,_=get_pose_by_index(i)
    print(pose)