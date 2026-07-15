# tests/test_motion_library.py —— 官方录制动作库【无损持久化】契约测试(TDD)
#
# 契约(C_LIB):把 104 个官方录制动作原样沉淀到磁盘,做到:
#   1. 发现:能从本机 HF 缓存找到 emotions/dances 两个库;
#   2. 无损:每个动作的原始 head 4×4 / antennas / body_yaw / time / check_collision 全部保留;
#   3. 便利:额外给出我们的 [T,9] 表示,且它与原始 head 可【无损往返】(证明分解不丢信息);
#   4. 神态:每条带 description(神态/适用情况)、时长、来源、许可,写入可读 index;
#   5. 完整:index 里每条都能在磁盘找到对应 npz。
# 若本机没有 HF 缓存(如 CI),相关用例 skip 而非 fail。
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from data_gen.build_motion_library import build_library, discover_moves, load_saved_move


def _require_cache():
    moves = discover_moves()
    if not moves:
        pytest.skip("本机未找到官方动作库 HF 缓存(需先启动过 Reachy Mini Control / sim daemon 触发下载)")
    return moves


def test_discover_finds_official_moves():
    moves = _require_cache()
    kinds = {m["kind"] for m in moves}
    assert "emotion" in kinds, "应至少发现情绪库"
    # 名字唯一
    names = [m["name"] for m in moves]
    assert len(names) == len(set(names)), "动作名不应重复"


def test_build_library_lossless_and_indexed(tmp_path):
    _require_cache()
    index = build_library(tmp_path, copy_audio=False)

    # 索引落地且自洽
    assert (tmp_path / "index.json").exists()
    assert (tmp_path / "index.md").exists()
    assert index["n_moves"] == len(index["moves"]) >= 1
    assert index["license"] == "apache-2.0"
    assert index["fps_original"] == 50
    assert len(index["dim_names"]) == 9

    got_nonempty_desc = False
    for entry in index["moves"]:
        npz = tmp_path / entry["npz"]
        assert npz.exists(), f"{entry['name']} 的 npz 缺失"
        d = np.load(npz)
        T = entry["n_frames"]
        assert T >= 2
        # 形状一致
        assert d["action9"].shape == (T, 9)
        assert d["head"].shape == (T, 4, 4)
        assert d["antennas"].shape == (T, 2)
        assert d["body_yaw"].shape == (T,)
        assert d["time"].shape == (T,)
        assert d["check_collision"].shape == (T,)
        # 数值健康
        assert not np.isnan(d["action9"]).any()
        assert d["action9"].dtype == np.float32
        # 时间从 0 起、单调不减(官方录制存在重复时间戳,无损保存原样保留)
        assert d["time"][0] == 0.0
        assert (np.diff(d["time"]) >= 0).all()
        # 神态/情况字段存在
        assert isinstance(entry["description"], str)
        got_nonempty_desc = got_nonempty_desc or bool(entry["description"].strip())

        # 真正的无损真值是原始 head(float64 原样存盘);action9 是派生便利视图。
        assert d["head"].dtype == np.float64
        # 往返:验证欧拉编码本身可逆——重建旋转应等于【正交化后的原始旋转】(官方矩阵
        # 非完美正交,R.from_matrix 会投影到最近旋转,故对照投影结果而非原始非正交阵)。
        a, H = d["action9"], d["head"]
        for k in (0, T // 2, T - 1):
            m = np.eye(4)
            m[:3, :3] = R.from_euler("xyz", a[k, 3:6]).as_matrix()
            m[:3, 3] = a[k, :3]
            R_ortho = R.from_matrix(H[k][:3, :3]).as_matrix()
            assert np.allclose(m[:3, 3], H[k][:3, 3], atol=1e-4), f"{entry['name']} 平移往返失配"
            assert np.allclose(m[:3, :3], R_ortho, atol=1e-5), f"{entry['name']} 旋转往返失配"
        # action9 的 body_yaw/antennas 应与原始逐帧一致
        assert np.allclose(a[:, 6], d["body_yaw"], atol=1e-5)
        assert np.allclose(a[:, 7], d["antennas"][:, 0], atol=1e-5)
        assert np.allclose(a[:, 8], d["antennas"][:, 1], atol=1e-5)

    assert got_nonempty_desc, "应至少有一条带非空 description(神态/情况)"


def test_load_saved_move_and_resample(tmp_path):
    _require_cache()
    index = build_library(tmp_path, copy_audio=False)
    name = index["moves"][0]["name"]

    # 原样加载:字段齐全,与 npz 一致
    m = load_saved_move(name, root=tmp_path)
    assert m["name"] == name and isinstance(m["description"], str)
    assert m["action9"].shape[1] == 9 and m["head"].ndim == 3

    # 重采样到 30Hz:时长基本保持、等步长、无 NaN
    r = load_saved_move(name, root=tmp_path, fps=30)
    dur = float(m["time"][-1])
    assert r["action9"].shape[0] == max(2, round(dur * 30))
    assert not np.isnan(r["action9"]).any()
    dt = np.diff(r["time"])
    assert np.allclose(dt, dt[0])  # 等步长
    # 重采样不改变原始真值 head(仍为原始帧数)
    assert r["head"].shape[0] == m["head"].shape[0]
