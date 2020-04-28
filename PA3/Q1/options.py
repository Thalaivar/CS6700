import copy

# 4 rooms
ROOM_1 = [[1,7], [2,7], [3,7], [4,7], [5,7],
            [1,8], [2,8], [3,8], [4,8], [5,8],
            [1,9], [2,9], [3,9], [4,9], [5,9],
            [1,10], [2,10], [3,10], [4,10], [5,10],
            [1,11], [2,11], [3,11], [4,11], [5,11]]

ROOM_2 = [[7,6], [8,6], [9,6], [10,6], [11,6],
            [7,7], [8,7], [9,7], [10,7], [11,7],
            [7,8], [8,8], [9,8], [10,8], [11,8],
            [7,9], [8,9], [9,9], [10,9], [11,9],
            [7,10], [8,10], [9,10], [10,10], [11,10],
            [7,11], [8,11], [9,11], [10,11], [11,11]]

ROOM_3 = [[7,1], [8,1], [9,1], [10,1], [11,1],
            [7,2], [8,2], [9,2], [10,2], [11,2],
            [7,3], [8,3], [9,3], [10,3], [11,3],
            [7,4], [8,4], [9,4], [10,4], [11,4]]

ROOM_4 = [[1,1], [2,1], [3,1], [4,1], [5,1],
            [1,2], [2,2], [3,2], [4,2], [5,2],
            [1,3], [2,3], [3,3], [4,3], [5,3],
            [1,4], [2,4], [3,4], [4,4], [5,4],
            [1,5], [2,5], [3,5], [4,5], [5,5]]
ROOMS = [ROOM_1, ROOM_2, ROOM_3, ROOM_4]

class HallwayOption:
    env = None

    def __init__(self, goal_type, room):
        if self.env is None:
            raise ValueError("Option environment not set")
        
        self.goal_type = goal_type
        self.goal = self.env.hallways[room-1][goal_type]
        self.create_init_set(goal_type, room)

    def create_init_set(self, goal_type, room):
        self.init_set = copy.deepcopy(ROOMS[room-1])
        if goal_type == 0:
            self.init_set.append(self.env.hallways[room-1][1])
        elif goal_type == 1:
            self.init_set.append(self.env.hallways[room-1][0])

    def policy(self, state):
        x, y = state
        # type 0 hallway -> walls on left and right
        if self.goal_type == 0:
            if x != self.goal[0]:
                if x > self.goal[0]:
                    action = 3
                elif x < self.goal[0]:
                    action = 2
            elif y != self.goal[1]:
                if y > self.goal[1]:
                    action = 1
                elif y < self.goal[1]:
                    action = 0
            
        elif self.goal_type == 1:
            if y != self.goal[1]:
                if y > self.goal[1]:
                    action = 1
                elif y < self.goal[1]:
                    action = 0

            elif x != self.goal[0]:
                if x > self.goal[0]:
                    action = 3
                elif x < self.goal[0]:
                    action = 2
        
        return action

    def run(self, state):
        # accumulated reward while executing option
        R = 0
        # time steps till option terminates
        k = 0
        option_on = True
        while option_on:
            if state not in self.init_set:
                option_on = False
                break
            else:
                # choose action according to option policy
                action = self.policy(state)
                # take action in environment
                state, r, done, _ = self.env.step(action)
                # accumulate reward
                R += (self.env.gamma**k)*r
                # update time steps
                option_on = not done
                k += 1

        return state, R, done, k