import heapq
from collections import deque

# 给定两个坐标，避开红方基地找最短路径
# from collections import deque
#
#
# def find_next_step(map, p1, p2):
#     x1, y1 = p1
#     x2, y2 = p2
#     rows, cols = len(map), len(map[0])
#     directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右四个方向
#
#     # 队列中存储 (当前坐标 x, 当前坐标 y, 当前路径)
#     queue = deque([(x1, y1, [(x1, y1)])])
#     visited = set()
#     visited.add((x1, y1))
#
#     while queue:
#         x, y, path = queue.popleft()
#
#         # 如果到达终点，返回路径中的第二个坐标
#         if (x, y) == (x2, y2):
#             if len(path) > 1:
#                 next_step = path[1]
#                 # 确保下一步不是敌方基地
#                 if map[next_step[0]][next_step[1]] != 2:
#                     return next_step
#             return x2, y2
#
#         for direction in directions:
#             nx, ny = x + direction[0], y + direction[1]
#             if (nx, ny) == (x2, y2):  # 如果下一步是目标位置就让其添加，不然永远返回none了
#                 visited.add((nx, ny))
#                 queue.append((nx, ny, path + [(nx, ny)]))
#
#             if 0 <= nx < rows and 0 <= ny < cols and (nx, ny) not in visited and map[nx][ny] != 2:
#                 visited.add((nx, ny))
#                 queue.append((nx, ny, path + [(nx, ny)]))
#
#     return None


import heapq


def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def find_next_step(map, p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    rows, cols = len(map), len(map[0])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右四个方向

    # 优先队列中存储 (优先级, 当前坐标 x, 当前坐标 y, 当前路径)
    open_list = [(0 + heuristic(p1, p2), x1, y1, [(x1, y1)])]
    heapq.heapify(open_list)
    visited = set()
    visited.add((x1, y1))
    path=[]

    while open_list:
        _, x, y, path = heapq.heappop(open_list)

        # 如果到达终点，返回路径中的第二个坐标
        if (x, y) == (x2, y2):
            if len(path) > 1:
                next_step = path[1]
                # 确保下一步不是敌方基地
                if map[next_step[0]][next_step[1]] != 2:
                    return next_step
            return (x2, y2)

        for direction in directions:
            nx, ny = x + direction[0], y + direction[1]
            if (nx, ny) == (x2, y2):
                visited.add((nx, ny))
                new_path = path + [(nx, ny)]
                priority = len(new_path) + heuristic((nx, ny), p2)
                heapq.heappush(open_list, (priority, nx, ny, new_path))

            if 0 <= nx < rows and 0 <= ny < cols and (nx, ny) not in visited and map[nx][ny] != 2:
                visited.add((nx, ny))
                new_path = path + [(nx, ny)]
                priority = len(new_path) + heuristic((nx, ny), p2)
                heapq.heappush(open_list, (priority, nx, ny, new_path))

    return path[1]


def find_shortest_distance(map, p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    rows, cols = len(map), len(map[0])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右四个方向

    # 队列中存储 (当前坐标 x, 当前坐标 y, 当前距离)
    queue = deque([(x1, y1, 0)])
    visited = set()
    visited.add((x1, y1))

    while queue:
        x, y, distance = queue.popleft()

        # 如果到达终点，返回当前距离
        if (x, y) == (x2, y2):
            return distance

        for direction in directions:
            nx, ny = x + direction[0], y + direction[1]

            if 0 <= nx < rows and 0 <= ny < cols and (nx, ny) not in visited and map[nx][ny] != 2:
                visited.add((nx, ny))
                queue.append((nx, ny, distance + 1))

    return 0

# class PathFinder:
#     def __init__(self, map_info):
#         self.map_info = map_info  # 初始化地图信息
#         self.n = len(map_info)  # 地图的行数
#         self.m = len(map_info[0])  # 地图的列数
#         # 定义四个方向的移动：上、下、左、右
#         self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
#         # 将方向映射为代码，便于路径重建时使用
#         self.direction_codes = {(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3}
#
#     def is_valid(self, position):
#         # 检查位置是否合法（在地图范围内且不是障碍物）
#         x, y = position
#         return 0 <= x < self.n and 0 <= y < self.m and self.map_info[x][y] != 2
#
#     def calculate_manhattan_distance(self, position1, position2):
#         # 计算曼哈顿距离，用作A*算法的启发函数
#         return abs(position1[0] - position2[0]) + abs(position1[1] - position2[1])
#
#     def a_star_next_step(self, start, goal):
#         # A*算法实现，只返回下一步方向
#         open_set = []
#         # 初始化open set并将起点加入优先队列
#         heapq.heappush(open_set, (0, start))
#         came_from = {}  # 用于路径重建
#         g_score = {start: 0}  # 起点到每个位置的实际代价
#         f_score = {start: self.calculate_manhattan_distance(start, goal)}  # 启发代价估计
#
#         while open_set:
#             # 从open set中取出f_score最小的节点
#             _, current = heapq.heappop(open_set)
#
#             if current == goal:
#                 # 如果当前节点是目标节点，则返回路径中的第二个节点方向
#                 return self.reconstruct_next_step_direction(came_from, current, start)
#
#             for direction in self.directions:
#                 # 计算邻居节点的位置
#                 neighbor = (current[0] + direction[0], current[1] + direction[1])
#
#                 if not self.is_valid(neighbor):
#                     # 如果邻居节点不合法（超出地图范围或是障碍物），跳过
#                     continue
#
#                 tentative_g_score = g_score[current] + 1  # 计算从起点到邻居节点的g_score
#
#                 if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
#                     # 如果发现更优的路径，则更新came_from和g_score
#                     came_from[neighbor] = (current, direction)
#                     g_score[neighbor] = tentative_g_score
#                     f_score[neighbor] = tentative_g_score + self.calculate_manhattan_distance(neighbor, goal)
#                     # 将邻居节点加入open set
#                     heapq.heappush(open_set, (f_score[neighbor], neighbor))
#
#         return None  # 如果未找到路径，返回None
#
#     def reconstruct_next_step_direction(self, came_from, current, start):
#         # 从目标节点回溯到起点，返回路径中的第二个节点方向
#         path = []
#         while current in came_from:
#             current, direction = came_from[current]
#             path.append(direction)
#             if current == start:
#                 break
#         path.reverse()  # 反转路径，使其从起点到终点
#         return self.direction_codes[path[0]] if len(path) > 0 else None  # 返回路径中的第二个节点方向
#
#     def find_next_step_to_base(self, start, base_positions):
#         # 在多个目标位置中寻找最短路径的下一步方向
#         min_step = None  # 保存最短路径的下一步方向
#         min_distance = float('inf')  # 初始化最短距离为无穷大
#         for base in base_positions:
#             # 对每一个目标位置使用A*算法寻找下一步方向
#             step = self.a_star_next_step(start, base)
#             if step is not None:
#                 # 计算从起点到目标位置的曼哈顿距离
#                 distance = self.calculate_manhattan_distance(start, base)
#                 if distance < min_distance:
#                     # 如果找到更短的路径，则更新最短路径的下一步方向和距离
#                     min_distance = distance
#                     min_step = step
#         return min_step  # 返回最短路径的下一步方向
#
#
# def test_a_star_next_step():
#     # 定义地图信息，0表示可通过，2表示障碍物
#     map_info = [
#         [0, 0, 0, 0, 0],
#         [0, 2, 2, 2, 0],
#         [0, 2, 0, 2, 0],
#         [0, 2, 2, 2, 0],
#         [0, 0, 0, 0, 0]
#     ]
#
#     # 创建PathFinder对象
#     path_finder = PathFinder(map_info)
#
#     # 定义起点和终点
#     start = (2, 2)
#     goal = (0, 0)
#
#     # 调用a_star_next_step函数
#     next_step = path_finder.a_star_next_step(start, goal)
#
#     # 打印结果
#     if next_step is not None:
#         print(f"Next step direction code from {start} to {goal} is: {next_step}")
#     else:
#         print(f"No valid path from {start} to {goal}.")


# if __name__ == '__main__':
#     # 运行测试函数
#     test_a_star_next_step()
