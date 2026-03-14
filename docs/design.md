# 2048-ai project
## 项目模块划分
1. core: 游戏的核心逻辑
2. ui: 游戏的界面逻辑
3. ai: 多种agent逻辑
4. test: 测试各部分的正确性

## 核心对象列表
1. Game
2. Board
3. Agent

## 每个对象的职责
1. Game
2. Board\
Board 内部使用指数表示法：0 表示空，1 表示 2，2 表示 4 ...\
Board.move(direction) 返回 (MoveStatus, score_gain)

## 关键接口
### core
1. self.step()\
输入：方向\
输出：StepStatue, info_dict\
职责：实现每一步direction，检查over情况，更新self
2. self.simulate_step()、
输入：方向\
输出：StepStatue, info_dict\
职责：与step相同，但是不会改变原self

## 游戏运行流程

## 模型训练流程

