import time
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def calculate_manhattan_distance(position1, position2):
    # 计算两个位置之间的曼哈顿距离
    return abs(position1[0] - position2[0]) + abs(position1[1] - position2[1])


class BattleFieldEnv:
    metadata = {'render.modes': ['console']}

    def __init__(self, map_layout, red_bases_info, blue_bases_info, fighters_info, cluster=0):
        super(BattleFieldEnv, self).__init__()
        self.map_layout = [list(row) for row in map_layout]  # 地图布局

        self.cluster = cluster  # 聚类信息

        self.red_bases = [{'position': pos, 'defense': defense, 'value': value, 'chose': True}  # true该基地可被选择
                          for pos, defense, value in red_bases_info]
        self.blue_bases = [{'position': pos, 'fuel': fuel, 'missile': missiles, 'chose': True}
                           for pos, fuel, missiles in blue_bases_info]
        self.fighters_info = [
            {'id': f_id, 'position': pos, 'max_fuel': max_fuel, 'max_missile': max_missiles, 'fuel': 0, 'missile': 0}
            for f_id, pos, max_fuel, max_missiles in fighters_info]
        self.blue_bases_positions = dict(
            (tuple(self.blue_bases[i]['position']), i) for i in range(len(self.blue_bases)))
        self.red_bases_positions = dict(
            (tuple(self.red_bases[i]['position']), i) for i in range(len(self.red_bases)))
        # 按fighter['position']计数
        self.fighters_positions = dict(
            (tuple(self.fighters_info[i]['position']), i) for i in range(len(self.fighters_info)))
        self._max_steps = 15000
        # n个战斗机的进攻列表
        self.stop = 0  # 如果有飞机弹尽粮绝则加1，所有飞机弹尽粮绝则停止程序
        self.is_ok = 0  # 执行有效信息才输出ok
        self.n_agents = len(fighters_info)
        self.attack_list = [0 for _ in range(self.n_agents)]
        self.pre_missil_target = [-1] * self.n_agents
        self.pre_fuel_target = [-1] * self.n_agents
        self.pre_attack_target = [-1] * self.n_agents  # 记录每一个战斗机的上一次攻击目标的索引
        self.n = len(self.map_layout)  # 地图行数
        self.m = len(self.map_layout[0])  # 地图列数

        self.direction_deltas = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # 上下左右
        self.reverse_direction_deltas = {v: k for k, v in self.direction_deltas.items()}
        self.fighter_paths = {}  # 每个飞机的移动路径对象
        self.calculate_path = np.zeros(self.n_agents)  # 是否计算路径？
        self.rewards = 0
        self.stop_incremented = [0] * self.n_agents
        pass

    def simulate(self, file=None):

        time_start = time.time()
        self.attack_list[0] = 1  # 暂时先让第一个飞机攻击
        i = 0
        step = 0
        reward_history = []  # 初始化奖励历史记录

        print("Simulation started")
        print(f"Initial attack list: {self.attack_list}")
        print(f"Initial red bases positions: {self.red_bases_positions}")
        print(f"Number of agents: {self.n_agents}")
        print(f"Maximum steps: {self._max_steps}")

        # Initialize tqdm progress bar
        progress_bar = tqdm(total=self._max_steps, desc="Simulation Progress", unit="step")

        progress_bar.set_description(f"Simulation Progress (Reward: {self.rewards:.2f})")

        while any(self.attack_list):

            # Check termination conditions
            if len(self.red_bases_positions) == 0:
                print("All red bases destroyed, ending simulation")
                break
            if self.stop >= self.n_agents:
                print(f"Stop condition met: {self.stop} >= {self.n_agents}, ending simulation")
                break
            if step > self._max_steps:
                print(f"Exceeded maximum steps: {step} > {self._max_steps}, ending simulation")
                break

            # Process the current agent's action if it's marked for attack
            if self.attack_list[i]:
                # 如果本架飞机本次有移动操作则可以切换到下一架飞机
                if self.process_fighter_action(i):
                    if i == self.n_agents - 1:
                        step += 1
                        reward_history.append(self.rewards)  # 记录当前总奖励
                        progress_bar.update(1)  # 更新进度条
                        progress_bar.set_description(f"Simulation Progress (Step: {step}, Reward: {self.rewards:.2f})")
                        if file:
                            file.write("OK\n")
                        self.is_ok = 0  # 所有飞机执行一次操作后一帧结束
                    self.attack_list[i] = 0  # 本飞机进攻标记置为0
                    i = (i + 1) % self.n_agents
                    self.attack_list[i] = 1  # 切换到下一架飞机

        progress_bar.close()  # Close the progress bar
        print("Simulation ended")
        time_end = time.time()
        print(f"Time elapsed: {time_end - time_start}")

        # 绘制奖励与步数之间的变化图
        plt.figure(figsize=(10, 5))
        plt.plot(reward_history, label="Reward per Step")
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.title("testcase5:Reward Progression Over Steps")
        plt.legend()
        plt.show()

    def process_fighter_action(self, i, file=None):
        from code.find_path import find_shortest_distance
        if self.fighters_info[i]['position'] in self.red_bases_positions:
            print("逆天，你是怎么飞上去的")
        move = False
        # 记录本架飞机本次有没有进行move操作，没有的话，下一次仍然由其进攻
        # 判断是否在我方基地上, 如果在则加油加弹
        on = self.on_blue_base(i)
        if on:  # 如果在我方基地上判断一下要加多少油，
            self._handle_refuel_action(i, file)
            self._handle_reload_action(i, file)
        # 计算到最近我方基地和最近敌方基地的Manhattan距离，估算是否回家
        a = self.get_blue_bases_with_sufficient_fuels(self.fighters_info[i]['position'])
        # 从当前位置回家的距离
        min_distance1 = find_shortest_distance(self.map_layout, self.fighters_info[i]['position'], a)
        # 计算目标基地位置
        b = self.get_closest_red_base(self.fighters_info[i]['position'])
        # 找到从目标基地到最近我方基地的坐标
        c = self.get_blue_bases_with_sufficient_fuels(b)
        # 计算从目标基地回家的距离
        min_distance2 = find_shortest_distance(self.map_layout, b, c)
        # 计算从目标基地到从目标基地到最近我方基地的距离

        # 没弹药就回家，因为你去哪都没用
        if self.fighters_info[i]['missile'] < 1:
            direction = self.back_for_missile(i)  # 这个坐标不应该是敌方基地，如果是则检查代码
            move = self._handle_move_action(i, direction, file)
            if self.on_blue_base(i):
                self._handle_refuel_action(i, file)
                self._handle_reload_action(i, file)
            # 如果正确执行了回家路径或者有弹药了则可以给下一架飞机操作了
            if move or self.fighters_info[i]['missile'] >= 1:
                # print(5)
                return True
            else:
                # 飞机有可能没油，学会放手，让下一架操作
                # print(4)
                return True
                # print("检查前面代码1")

        # 如果有弹药，检查燃油是否足够去进攻
        # self.fighters_info[0]['missile']
        elif min_distance1 == self.fighters_info[i]['fuel'] or min_distance1 + 1 == self.fighters_info[i]['fuel'] \
                or self.fighters_info[i]['fuel'] <= min_distance1 + min_distance2:
            direction = self.back_for_fuel(i)  # 这个坐标不应该是敌方基地，如果是则检查代码
            move = self._handle_move_action(i, direction, file)
            if self.on_blue_base(i):
                self._handle_reload_action(i, file)
                self._handle_refuel_action(i, file)

            # 在这里可以加进攻逻辑，回去路上可以顺手一打
            action_id, position = self.check_surroundings(i)
            bre = 0
            while action_id and bre < 3:  # 这里主要是为了尽可能地把周围可能的敌方基地都打一下
                bre += 1
                self.attack_target(i, position)  # 既然曼哈顿距离为1，那么不用移动直接攻击
                action_id, position = self.check_surroundings(i)
            if not move:
                return True
            else:
                # print(3)
                return True
        else:  # 如果不是在回家的路上
            # 检查周围环境，判断下一步动作
            action_id, position = self.check_surroundings(i)
            b = 0
            while action_id and b < 3:  # 这里主要是为了尽可能地把周围可能的敌方基地都打一下
                b += 1
                self.attack_target(i, position)  # 既然曼哈顿距离为1，那么不用移动直接攻击
                action_id, position = self.check_surroundings(i)
            # 如果周围没有敌方基地了或者你弹药没了，就向下一个敌方基地进攻他，下一轮再判断回家还是进攻
            direction = self.go_to_attack(i)  # 这个direction可能会直接返回飞机的坐标
            # 如果敌方基地就在附近则可能返回飞机的坐标，这个时候直接结束本次操作

            if direction != -1:  # 如果有可达的敌方基地
                if direction in self.red_bases_positions:
                    # 即下一个坐标是敌方基地,那就打，看能不能打掉，如果不能打掉
                    # 交给下一架处理，如果能打掉就走该基地
                    self.attack_target(i, direction)
                    if direction in self.red_bases_positions:
                        return True  # 下一个坐标还是敌方基地，那么还是让下一架飞机先处理吧
                    # 如果基地被打掉了，那就能走了
                    elif self._handle_move_action(i, direction, file):
                        move = True
                # 如果下一位置不是敌方基地，那就直接走
                elif self._handle_move_action(i, direction, file):
                    move = True
                else:
                    # 飞机有可能没油了，让下一架飞机去操作
                    # print(i)
                    return True
                    # print("检查前面代码3")
            else:
                # print(i)  # 想一下如果没有目标基地了，检查代码
                return True  # 如果本架飞机没有进攻目标了，则给下一个飞机操作

        return move

    def back_for_fuel(self, i):
        from code.find_path import find_next_step
        dre = -1  # 以下代码不能正常执行则返回-1以便调试

        if self.cluster == 3 or self.cluster == 4:
            from Kmeans import assign_targets_to_fighters

            cluster_centers, assignments = assign_targets_to_fighters(self.red_bases_positions, \
                                                                      self.fighters_info, self.n_agents)
            target_position = cluster_centers[assignments[i]]
            # 转换为元组
            target_position = tuple(target_position)
            if target_position in self.red_bases_positions:
                target_position = self.get_blue_bases_with_sufficient_fuels(self.fighters_info[i]['position'])
            if target_position == self.fighters_info[i]['position']:
                target_position = self.get_blue_bases_with_sufficient_fuels(self.fighters_info[i]['position'])
            if target_position is not None:
                dre = find_next_step(self.map_layout, self.fighters_info[i]['position'],
                                     target_position)
        else:
            target_position = self.get_blue_bases_with_sufficient_fuels(self.fighters_info[i]['position'])

            if target_position is not None:
                dre = find_next_step(self.map_layout, self.fighters_info[i]['position'],
                                     target_position)

        # 是否给基地加锁？目前发现不加效果更好
        # if target_position in self.blue_bases_positions:
        #     target_index = self.blue_bases_positions[target_position]
        #     pre_target = self.pre_fuel_target[i]
        #     if target_index != pre_target:  # 如果改变了攻击目标
        #         if pre_target != -1:  # 初始化没有选择目标，都为-1
        #             self.blue_bases[pre_target]['chose'] = True  # 先把上一目标解锁
        #             self.pre_fuel_target[i] = target_index
        #             # 记录该基地已被选择，下一架飞机要进攻时不要选此基地
        #             self.blue_bases[target_index]['chose'] = False
        #         else:
        #             self.pre_fuel_target[i] = target_index
        #             self.blue_bases[target_index]['chose'] = False

        if dre is None or dre == -1:
            return -1
        else:
            return dre

    def go_to_attack(self, i):
        from code.find_path import find_next_step
        dre = -1
        if self.cluster == 1 or self.cluster == 4:
            from Kmeans import assign_targets_to_fighters
            cluster_centers, assignments = assign_targets_to_fighters(self.red_bases_positions,
                                                                      self.fighters_info, self.n_agents)

            target_position = cluster_centers[assignments[i]]
            # 转换为元组
            target_position = tuple(target_position)

            if target_position == self.fighters_info[i]['position']:
                target_position = self.get_closest_red_base(self.fighters_info[i]['position'])
            if target_position is not None:
                dre = find_next_step(self.map_layout, self.fighters_info[i]['position'],
                                     target_position)

        else:
            target_position = self.get_closest_red_base(self.fighters_info[i]['position'])
            if target_position is not None:
                dre = find_next_step(self.map_layout, self.fighters_info[i]['position'],
                                     target_position)

        # if target_position in self.red_bases_positions:
        #     target_index = self.red_bases_positions[target_position]
        #     pre_target = self.pre_attack_target[i]
        #     if target_index != pre_target:  # 如果改变了攻击目标
        #         if pre_target != -1:  # 初始化没有选择目标，都为-1
        #             self.red_bases[pre_target]['chose'] = True  # 先把上一目标解锁
        #             self.pre_attack_target[i] = target_index
        #             # 记录该基地已被选择，下一架飞机要进攻时不要选此基地
        #             self.red_bases[target_index]['chose'] = False
        #         else:
        #             self.pre_attack_target[i] = target_index
        #             # 记录该基地已被选择，下一架飞机要进攻时不要选此基地
        #             self.red_bases[target_index]['chose'] = False
        if dre is None or dre == -1:
            # self.red_bases_positions
            return -1
        else:
            return dre

    def back_for_missile(self, i):
        from code.find_path import find_next_step
        # 每次都重新规划路径，因为之前不能走的敌方基地被摧毁后就可能又可以走了
        dre = -1


        if self.cluster == 2 or self.cluster == 4:
            from Kmeans import assign_targets_to_fighters
            # target_position = self.get_closest_red_base(self.fighters_info[i]['position'])

            cluster_centers, assignments = assign_targets_to_fighters(self.red_bases_positions, \
                                                                      self.fighters_info, self.n_agents)

            target_position = cluster_centers[assignments[i]]
            # 转换为元组

            target_position = tuple(target_position)

            if target_position in self.red_bases_positions:
                target_position = self.get_blue_bases_with_sufficient_missiles(self.fighters_info[i]['position'])

            if target_position == self.fighters_info[i]['position']:
                target_position = self.get_blue_bases_with_sufficient_missiles(self.fighters_info[i]['position'])
            if target_position is not None:
                dre = find_next_step(self.map_layout, self.fighters_info[i]['position'],
                                     target_position)
        else:
            target_position = self.get_blue_bases_with_sufficient_missiles(self.fighters_info[i]['position'])
            if target_position is not None:
                dre = find_next_step(self.map_layout, self.fighters_info[i]['position'],
                                     target_position)
        # if target_position in self.blue_bases_positions:
        #     target_index = self.blue_bases_positions[target_position]
        #     pre_target = self.pre_missil_target[i]
        #     if target_index != pre_target:  # 如果改变了攻击目标
        #         if pre_target != -1:  # 初始化没有选择目标，都为-1
        #             self.blue_bases[pre_target]['chose'] = True  # 先把上一目标解锁
        #             self.pre_missil_target[i] = target_index
        #             # 记录该基地已被选择，下一架飞机要进攻时不要选此基地
        #             self.blue_bases[target_index]['chose'] = False
        #         else:
        #             self.pre_missil_target[i] = target_index
        #             # 记录该基地已被选择，下一架飞机要进攻时不要选此基地
        #             self.blue_bases[target_index]['chose'] = False

        if dre is None or dre == -1:
            print(dre)
            return -1
        else:
            return dre

    def get_closest_red_base(self, current_plane_position):
        # 初始化一个列表来存储未经摧毁且未被选择的敌方基地
        target_bases = []
        # 遍历所有红方基地
        for base_position, base_id in self.red_bases_positions.items():
            # 检查该基地的防御值是否大于0且未被选择
            if self.red_bases[base_id]['defense'] >= 0 and self.red_bases[base_id]['chose']:
                # 如果是，将基地坐标添加到列表中
                target_bases.append(base_position)

        # 如果没有符合条件的基地，返回None和无穷大距离
        if not target_bases:
            return None

        # 根据曼哈顿距离排序
        sorted_bases_by_distance = sorted(
            target_bases,
            key=lambda base: abs(base[0] - current_plane_position[0]) + abs(base[1] - current_plane_position[1])
        )

        if len(sorted_bases_by_distance) >= 10:  # 如果满足的基地数量大于10，则只要前面十个
            sorted_bases_by_distance = sorted_bases_by_distance[:10]

        # 根据军事价值对基地进行排序
        sorted_bases = sorted(
            sorted_bases_by_distance,
            key=lambda base: self.red_bases[base_id]['value'],
            reverse=True
        )
        return sorted_bases[0]

    def get_blue_bases_with_sufficient_missiles(self, current_plane_position):
        # 初始化一个列表来存储弹药储备大于0的蓝方基地坐标
        sufficient_missile_bases = []
        # 遍历所有蓝方基地
        for base_position, base_id in self.blue_bases_positions.items():
            # 检查该基地的弹药储备是否大于0并且被选中
            if self.blue_bases[base_id]['missile'] >= 1 and self.blue_bases[base_id]['chose']:
                # 如果是，将基地坐标添加到列表中
                sufficient_missile_bases.append(base_position)

        # 如果没有符合条件的基地，返回 None 和 None
        if not sufficient_missile_bases:
            return None

        sorted_bases_by_distance = sorted(
            sufficient_missile_bases,
            key=lambda base: abs(base[0] - current_plane_position[0]) + abs(base[1] - current_plane_position[1])
        )

        if len(sorted_bases_by_distance) >= 10:  # 如果满足的基地数量大于10，则只要前面十个
            sorted_bases_by_distance = sorted_bases_by_distance[:10]

        # 根据弹量对基地进行排序
        sorted_bases = sorted(
            sorted_bases_by_distance,
            key=lambda base: self.blue_bases[base_id]['missile'],
            reverse=True
        )

        # 找到近的弹药多的蓝方基地
        closest_base = sorted_bases[0]

        # 返回最近的蓝方基地坐标
        return closest_base

    def get_blue_bases_with_sufficient_fuels(self, current_plane_position):
        # 初始化一个列表来存储燃料储备大于10且被选中的蓝方基地坐标
        # current_plane_position = self.fighters_info[i]['position']
        sufficient_fuel_bases = []
        # 遍历所有蓝方基地

        for base_position, base_info in self.blue_bases_positions.items():
            # 检查该基地的燃料储备是否大于0且被选中
            if self.blue_bases[base_info]['fuel'] > 0 and self.blue_bases[base_info]['chose']:
                # 如果是，将基地坐标添加到列表中
                sufficient_fuel_bases.append(base_position)

        # 如果没有符合条件的基地，返回 None
        if not sufficient_fuel_bases:
            return None

        # 根据曼哈顿距离对基地进行排序
        sorted_bases_by_distance = sorted(
            sufficient_fuel_bases,
            key=lambda base: abs(base[0] - current_plane_position[0]) + abs(base[1] - current_plane_position[1])
        )

        if len(sorted_bases_by_distance) >= 10:  # 如果满足的基地数量大于10，则只要前面十个
            sorted_bases_by_distance = sorted_bases_by_distance[:10]

        # 根据油量对基地进行排序
        sorted_bases = sorted(
            sorted_bases_by_distance,
            key=lambda base: self.blue_bases[base_info]['fuel'],
            reverse=True
        )

        # 找到最近的蓝方基地
        closest_base = sorted_bases[0]

        # 返回最近的蓝方基地坐标和最短距离
        return closest_base

    def check_surroundings(self, fighter_id):
        fighter_position = self.fighters_info[fighter_id]['position']
        # 检查周围1个曼哈顿距离内是否有红方基地
        for red_base in self.red_bases:
            if calculate_manhattan_distance(fighter_position, red_base['position']) == 1 \
                    and red_base['defense'] >= 0:
                return 1, red_base['position']

        # 如果不满足以上条件，返回0和空
        return 0, None

    def calculate_info_to_base(self, fighter_position, base_position):
        # 地图以左上角为起点，坐标位置第一个是纵坐标，第二个是横坐标
        fighter_y, fighter_x = fighter_position
        base_y, base_x = base_position

        # 初始化进攻信息列表，第一个参数决定上下方向，第二个参数是纵向距离，第三个参数是左右方向，第四个参数是横向距离
        attack_info = [-1, -1, -1, -1]

        # 计算纵向距离和方向
        if fighter_y > base_y:
            attack_info[0] = 0  # 上
            attack_info[1] = fighter_y - base_y
        else:
            attack_info[0] = 1  # 下
            attack_info[1] = base_y - fighter_y

        # 计算横向距离和方向
        if fighter_x > base_x:
            attack_info[2] = 2  # 左
            attack_info[3] = fighter_x - base_x
        else:
            attack_info[2] = 3  # 右
            attack_info[3] = base_x - fighter_x

        return attack_info

    def attack_target(self, fighter_id, target_position, file=None):
        h, w = target_position
        if (h, w) in self.red_bases_positions:
            target_base_id = self.red_bases_positions[(h, w)]
        else:
            # 处理键不存在的情况,case3有问题，先注释了
            # print("Key", {h, w}, " not found in red_bases_positions")
            target_base_id = None  # 或者进行其他处理

        attack_info = self.calculate_info_to_base(self.fighters_info[fighter_id]['position'], target_position)
        up_dre, ud_d, lr_dre, lr_d = attack_info

        if lr_d == 0:
            self._handle_attack_action(fighter_id, up_dre, file)
        elif ud_d == 0:
            self._handle_attack_action(fighter_id, lr_dre, file)
        else:
            print("不应该到这里，请检查前面的代码")

    def on_blue_base(self, fighter_id):
        x, y = self.fighters_info[fighter_id]['position']
        if (x, y) in self.blue_bases_positions:
            # 如果位于我方基地上
            return True
        else:
            # print(f"Fighter {fighter_id} at coordinates ({x}, {y}) is not on a base.")
            return False

    def is_valid_move(self, new_x, new_y, i):
        return (0 <= new_x < self.n and 0 <= new_y < self.m and
                self.fighters_info[i]['fuel'] >= 1 and
                self.map_layout[new_x][new_y] != 2)

    def move_fighter(self, new_x, new_y, i, file=None):
        current_position = self.fighters_info[i]['position']
        delta = (new_x - current_position[0], new_y - current_position[1])
        dir = self.reverse_direction_deltas.get(delta, None)

        if dir is not None:
            self.fighters_info[i]['position'] = (new_x, new_y)
            self.fighters_info[i]['fuel'] -= 1

            if file:
                # print(f'move {self.fighters_info[i]["id"]} {dir}\n')
                file.write(f'move {self.fighters_info[i]["id"]} {dir}\n')
                return True
        else:
            self.fighters_info[0]['position']
            print(1)

        return False

    def _handle_move_action(self, i, direction, file=None):

        # 当前战斗机的位置
        # 计算新位置
        new_x, new_y = direction
        if self.is_valid_move(new_x, new_y, i):
            return self.move_fighter(new_x, new_y, i, file)
        else:
            if not (0 <= new_x < self.n and 0 <= new_y < self.m):
                print(
                    f"Error: Move out of bounds for fighter {self.fighters_info[i]['id']} trying to move to ({new_x}, {new_y}).")
            elif self.fighters_info[i]['fuel'] < 1 and not self.stop_incremented[i]:
                self.stop = self.stop + 1
                self.stop_incremented[i] = 1  # 保证每架战斗机最多只能死一次
                print(
                    f"Error: Not enough fuel for fighter {self.fighters_info[i]['id']} to move. Current fuel: {self.fighters_info[i]['fuel']}.")
            elif self.map_layout[new_x][new_y] == 2:
                print("Can't pass a non-destroyed red base")

            return False

    def _handle_attack_action(self, i, attack_direction, file=None):
        delta = self.direction_deltas[attack_direction]
        target_position = (
            self.fighters_info[i]['position'][0] + delta[0], self.fighters_info[i]['position'][1] + delta[1])

        if target_position in self.red_bases_positions:
            target_base_id = self.red_bases_positions[target_position]
            missile_count = min(self.fighters_info[i]['missile'], self.red_bases[target_base_id]['defense'])
            if self.fighters_info[i]['missile'] < missile_count:
                raise ValueError(
                    f"Error: Fighter {self.fighters_info[i]['id']} does not have enough missiles for the attack.")
            elif missile_count >= 0:
                self.fighters_info[i]['missile'] -= missile_count
                self.red_bases[target_base_id]['defense'] -= missile_count
                if file:
                    file.write(f'attack {self.fighters_info[i]["id"]} {attack_direction} {missile_count}\n')
                    # print(f'attack {self.fighters_info[i]["id"]} {attack_direction} {missile_count}\n')
                if self.red_bases[target_base_id]['defense'] <= 0:
                    self.rewards += self.red_bases[target_base_id]['value']
                    del self.red_bases_positions[target_position]
                    x, y = target_position
                    self.map_layout[x][y] = 0
                    # print(
                    #     f"Base {target_base_id} at position {target_position} destroyed by fighter {self.fighters_info[i]['id']}.")
                else:
                    a = 1
                    # print(
                    #     f"Fighter {self.fighters_info[i]['id']} attacked base {target_base_id} \
                    #     at position {target_position} with\
                    #      {missile_count} missiles. Base defense reduced to \
                    #      {self.red_bases[target_base_id]['defense']}.")


        else:
            a = 1
            # case3有防御值为0的敌方基地，这里先注释掉
            # print(f"Error: No valid target at position {target_position} for fighter {self.fighters_info[i]['id']}.")

    def _handle_refuel_action(self, fighter_id, file=None):
        current_pos = self.fighters_info[fighter_id]['position']
        if current_pos in self.blue_bases_positions:
            base_id = self.blue_bases_positions[current_pos]
            # 计算需要添加的燃料和实际可添加的燃料
            add_fuels = self.fighters_info[fighter_id]['max_fuel'] - self.fighters_info[fighter_id]['fuel']
            actual_fuel = min(add_fuels, self.blue_bases[base_id]['fuel'])

            if actual_fuel > 0:
                self.fighters_info[fighter_id]['fuel'] += actual_fuel
                self.blue_bases[base_id]['fuel'] -= actual_fuel
                if file:
                    file.write(f'fuel {self.fighters_info[fighter_id]["id"]} {actual_fuel}\n')
                    # print('fuel {self.fighters_info[fighter_id]["id"]} {actual_fuel}\n')
        else:
            # print(f"Error: Fighter {self.fighters_info[i]['id']} is not at a valid base position.")
            return

    def _handle_reload_action(self, fighter_id, file=None):
        current_pos = self.fighters_info[fighter_id]['position']
        if current_pos in self.blue_bases_positions:
            base_id = self.blue_bases_positions[current_pos]
            # 计算需要添加的燃料和实际可添加的燃料
            add_missile = self.fighters_info[fighter_id]['max_missile'] - self.fighters_info[fighter_id]['missile']
            actual_missile = min(add_missile, self.blue_bases[base_id]['missile'])
            if actual_missile > 0:
                self.fighters_info[fighter_id]['missile'] += actual_missile
                self.blue_bases[base_id]['missile'] -= actual_missile
                if file:
                    # print(f'missile {self.fighters_info[fighter_id]["id"]} {actual_missile}\n')
                    file.write(f'missile {self.fighters_info[fighter_id]["id"]} {actual_missile}\n')
        else:
            a = 1
            # print(f"Error: Fighter {self.fighters_info[i]['id']} is not at a valid base position.")
