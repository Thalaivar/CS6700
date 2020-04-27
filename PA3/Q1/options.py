class HallwayOption:
    def __init__(self, I, goal_state, goal_type, env, other_hallway):
        # initiation set
        self.init_set = I
        self.init_set.append(other_hallway)
        # state agent should reach
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
        # hallway state with walls on left and right
        elif self.goal_state == 1:
            # align horizontally with hallway first
            if x != self.goal_state[0]:
                if x > self.goal_state[0]:
                    action = 3
                if x < self.goal_state[0]:
                    action = 2
                
            # if aligned horizontally, move toward hallway
            elif y != self.goal_state[1]:
                if y > self.goal_state[1]:
                    action = 1
                elif y < self.goal_state[1]:
                    action = 0

        return action

    # run option from current state to termination
    def run(self, state):
        R = 0
        k = 0
        option_on = True
        while option_on:
            # terminate if state belongs to termination set
            if state not in self.init_set:
                option_on = False
                break
            else:
                # take action according to policy
                action = self.policy(state)
                # one step reward
                state, r, done, _ = self.env.step(action)
                # accumulate reward
                R += (self.env.gamma**k)*r
                # terminate option if task is done
                option_on = not done
                k += 1

        return state, R, done