class HallwayOption:
    def __init__(self, I, beta, goal_state, goal_type, env):
        self.init_set = I
        self.term_set = beta
        self.goal_state = goal_state
        self.goal_type = goal_type
        self.env = env

    def policy(self, state):
        x, y = state
        # hallway state with walls on top and bottom
        if self.goal_type == 0:
            # align vertically with hallway first
            if y != self.goal_state[1]:
                if y > self.goal_state[1]:
                    action = 1
                elif y < self.goal_state[1]:
                    action = 0
            # if aligned vecrtically, move toward hallway
            elif x != self.goal_state[0]:
                if x > self.goal_state[0]:
                    action = 3
                elif x < self.goal_state[0]:
                    action = 2

        return action

    def run(self, state):
        r = 0
        option_on = True:
        while option_on:
            if state in self.term_set:
                option_on = False
            else:
                