import numpy as np
from code.combat import BattleFieldEnv


def create_env_from_info(info_dict, cluster, BattleFieldEnv=None):
    # 确保'map_layout'键存在并获取其值
    map_layout = info_dict.get('map_layout')
    # print(map_layout[0][6])

    # 处理'blue_bases'信息，确保'attributes'包含至少两个元素
    blue_bases_info = [
        (base['position'], base['attributes'][0], base['attributes'][1])
        for base in info_dict.get('blue_bases', [])
        if len(base.get('attributes', [])) >= 2
    ]

    # 处理'red_bases'信息，确保'attributes'包含至少四个元素
    red_bases_info = [
        (base['position'], base['attributes'][2], base['attributes'][3])
        for base in info_dict.get('red_bases', [])
        if len(base.get('attributes', [])) >= 4
    ]

    # 处理fighters信息，确保每个fighter包含至少四个元素
    fighters_info = [
        (idx, tuple(fighter[:2]), fighter[2], fighter[3])
        for idx, fighter in enumerate(info_dict.get('fighters', []))
        if len(fighter) >= 4
    ]

    # 创建并返回BattleFieldEnv实例
    e = BattleFieldEnv(
        map_layout=map_layout,
        red_bases_info=red_bases_info,
        blue_bases_info=blue_bases_info,
        fighters_info=fighters_info,
        cluster=cluster
    )

    return e


def parse_game_data_from_file(filename):
    with open(filename, 'r') as file:
        lines = file.read().strip().split('\n')
        # print(lines)
        n, m = map(int, lines[0].split())  # 地图大小
        map_layout = lines[1:n + 1]  # 地图布局
        # print(map_layout)
        offset = n + 1
        blue_base_count = int(lines[offset])  # 蓝方基地数量
        blue_bases = []
        for i in range(offset + 1, offset + 1 + 2 * blue_base_count, 2):
            if i + 1 >= len(lines):  # 检查是否超出索引范围
                print("错误：基地信息不完整。")
                return None
            position = tuple(map(int, lines[i].split()))  # 基地位置
            attributes = list(map(int, lines[i + 1].split()))  # 基地属性
            blue_bases.append({'position': position, 'attributes': attributes})

        offset += 2 * blue_base_count + 1
        red_base_count = int(lines[offset])  # 红方基地数量
        red_bases = []
        for i in range(offset + 1, offset + 1 + 2 * red_base_count, 2):
            if i + 1 >= len(lines):  # 检查是否超出索引范围
                print("错误：基地信息不完整。")
                return None
            position = tuple(map(int, lines[i].split()))  # 基地位置
            attributes = list(map(int, lines[i + 1].split()))  # 基地属性
            red_bases.append({'position': position, 'attributes': attributes})

        offset += 2 * red_base_count + 1
        if offset >= len(lines):  # 检查战斗机数量行是否存在
            print("错误：缺少战斗机数量信息。")
            return None
        fighter_count = int(lines[offset])  # 战斗机数量
        fighters = []
        for i in range(offset + 1, offset + 1 + fighter_count):
            if i >= len(lines):  # 检查是否超出索引范围
                print("错误：战斗机信息不完整。")
                return None
            fighters.append(list(map(int, lines[i].split())))  # 战斗机属性

        map_observation = np.zeros((len(map_layout), len(map_layout[0])), dtype=np.uint8)
        for i, row in enumerate(map_layout):
            for j, cell in enumerate(row):
                if cell == '.':
                    map_observation[i, j] = 0  # 用0表示空白区域
                elif cell == '#':
                    map_observation[i, j] = 2  # 用1表示红方基地图
                elif cell == '*':
                    map_observation[i, j] = 1  # 用2表示蓝方基地图
        map_layout = map_observation

        return {
            'map_size': (n, m),
            'map_layout': map_layout,
            'blue_bases': blue_bases,
            'red_bases': red_bases,
            'fighters': fighters
        }


if __name__ == '__main__':
    filename_input = "data/input/testcase1.in"  # 训练地图
    info_dict = parse_game_data_from_file(filename_input)  # 解析地图
    # cluster=0表示不聚类，1表示对敌方基地聚类，2表示加弹时对我方基地聚类，3表示加油时对我方基地聚类,4表示对敌我双方基地聚类
    cluster = 0
    env = create_env_from_info(info_dict, cluster, BattleFieldEnv)  # 创建环境
    filename_output = "data/output/output1.txt"  # 指令文件
    with open(filename_output, "w") as file:
        env.simulate(file)
