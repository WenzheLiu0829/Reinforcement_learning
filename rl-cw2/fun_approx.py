import cv2
import sys
import numpy as np

from enduro.agent import Agent
from enduro.action import Action
import matplotlib.pyplot as plt

FEATURE_NUM = 10
ACTIONS = [Action.ACCELERATE,Action.LEFT,Action.RIGHT,Action.BRAKE]
learning_rate = 0.0001
decay_rate = 0.9
reward_List = [None]*500
weight_List = np.zeros((500,10))

class FunctionApproximationAgent(Agent):
    def __init__(self):
        super(FunctionApproximationAgent, self).__init__()
        # Add member variables to your class here
        self.total_reward = 0
        self.reward = 0
        self.Q_target = 0
        self.Q_predict = 0
        self.weights = np.random.uniform(-1,1,FEATURE_NUM)
        self.features = np.zeros(FEATURE_NUM)
        self.action = Action.ACCELERATE
        self.count = -1
        self.index = 0
        self.epsilon = 0.01
    def initialise(self, road, cars, speed, grid):
        """ Called at the beginning of an episode. Use it to construct
        the initial state.

        Args:
            road  -- 2-dimensional array containing [x, y] points
                     in pixel coordinates of the road grid
            cars  -- dictionary which contains the location and the size
                     of the agent and the opponents in pixel coordinates
            speed -- the relative speed of the agent with respect the others
            gird  -- 2-dimensional numpy array containing the latest grid
                     representation of the environment

        For more information on the arguments have a look at the README.md
        """

        # Reset the total reward for the episode
        self.total_reward = 0
        self.reward = 0
        self.Q_target = 0
        self.Q_predict = 0
        self.count += 1
        self.state = (road, cars, speed, grid)
        self.index = 0
        self.epsilon = 0.01
    def act(self):
        """ Implements the decision making process for selecting
        an action. Remember to store the obtained reward.
        """
        # You can get the set of possible actions and print them with:
        # print [Action.toString(a) for a in self.getActionsSet()]

        # Execute the action and get the received reward signal
        if np.random.uniform(0., 1.) < self.epsilon:
            self.action = ACTIONS[np.random.choice(4)]
        else:
            self.action = ACTIONS[self.index]

        self.reward = self.move(self.action)
        self.total_reward += self.reward

        # IMPORTANT NOTE:
        # 'action' must be one of the values in the actions set,
        # i.e. Action.LEFT, Action.RIGHT, Action.ACCELERATE or Action.BRAKE
        # Do not use plain integers between 0 - 3 as it will not work
    

    def sense(self, road, cars, speed, grid):
        """ Constructs the next state from sensory signals.

        Args:
            road  -- 2-dimensional array containing [x, y] points
                     in pixel coordinates of the road grid
            cars  -- dictionary which contains the location and the size
                     of the agent and the opponents in pixel coordinates
            speed -- the relative speed of the agent with respect the others
            gird  -- 2-dimensional numpy array containing the latest grid
                     representation of the environment

        For more information on the arguments have a look at the README.md
        """
        self.next_state = (road, cars, speed, grid)

        
    def learn(self):
        """ Performs the learning procedure. It is called after act() and
        sense() so you have access to the latest tuple (s, s', a, r).
        """
        self.Q_target = np.dot(self.weights, self.features)

        temp_Q_predict = []
        temp_features = []
        for action in ACTIONS:    
            features = self.buildFeatures(self.state, self.next_state, action)
            temp_features.append(features)
            temp_Q_predict.append(np.dot(self.weights, features))

        self.index = np.argmax(temp_Q_predict)
        
        self.Q_predict = temp_Q_predict[self.index]

        self.features = temp_features[self.index]


        error = self.reward + decay_rate*self.Q_predict - self.Q_target
        self.weights += learning_rate * error * self.features

        self.state = self.next_state
  

    def feature_1(self, next_state, action):
        '''
        1 - if the agent is in the center of the road and take accelerate action
        0 - otherwise
        '''

        grid = next_state[3]
        [[agent_grid_col]] = np.argwhere(grid[0, :] == 2)
        if agent_grid_col == 4 or agent_grid_col == 5:
            if action == Action.ACCELERATE:
                return True
            else:
                return False
        else:
            return False

    def feature_2(self, next_state, action):
        '''
        1 - if the agent doesn't collide with the opponents and take accelerate action
        0 - otherwise
        '''
        cars = next_state[1]
        if not cars['others']:
            return False

        x, y, w, h = cars['self']
        
        min_dist = sys.float_info.max
        min_angle = 0.

        for c in cars['others']:
            cx, cy, _, _ = c
            dist = np.sqrt((cx - x) ** 2 + (cy - y) ** 2)
            if dist < min_dist:
                min_dist = dist
                min_angle = np.arctan2(y - cy, cx - x)

        collision = min_dist < 18. and 0.1 * np.pi < min_angle and min_angle < 0.9 * np.pi
        
        if (collision == False and action == Action.ACCELERATE):
            return True
        else:
            return False

    def feature_3(self, state, next_state):
        '''
        1 - if the speed of next state is faster than the current state
        0 - otherwise
        '''
        speed = state[2]
        next_speed = next_state[2]
        if next_speed > speed:
            return True
        else:
            return False

    def feature_4(self, next_state, action):
        '''
        1 - if there is no opponent in the road and take accelerate action
        0 - otherwise
        '''
        cars = next_state[1]
        opponent_road = cars['others']
        opponent_num = len(opponent_road)
        if (opponent_num == 0 and action == Action.ACCELERATE):
            return True
        else:
            return False

    def feature_5(self, next_state, action):
        '''
        1 - if there is an opponent in front of you and take the correclt action to avoid it
        0 - otherwise
        '''
        state = [0, 0]
        grid = next_state[3]
        # Agent position (assumes the agent is always on row 0)
        [[x]] = np.argwhere(grid[0, :] == 2)
        state[0] = x

        # Sum the rows of the grid
        rows = np.sum(grid, axis=1)
        # Ignore the agent
        rows[0] -= 2
        # Get the closest row where an opponent is present
        rows = np.sort(np.argwhere(rows > 0).flatten())

        # If any opponent is present
        if rows.size > 0:
            # Add the x position of the first opponent on the closest row
            row = rows[0]
            for i, g in enumerate(grid[row, :]):
                if g == 1:     
                    # 0 means that no agent is present and so
                    # the index is offset by 1
                    state[1] = i + 1
                    break

        agent_col = state[0]
        opponent_col = state[1]
        # if the agent is not in the both side of the road and there is an opponent in front of the agent
        if (agent_col != 1 and agent_col != 8) and (agent_col == opponent_col):
            if (action == Action.LEFT or action == Action.RIGHT):
                return True
            else:
                return False
        elif (agent_col == 1 and action == Action.RIGHT and agent_col == opponent_col):
            return True
        elif (agent_col == 8 and action == Action.LEFT and agent_col == opponent_col):
            return True
        else:
            return False

    def feature_6(self, next_state, action):
        '''
        1 - if action a would most probably take the agent into the wall
        0 - otherwise
        '''
        grid = next_state[3]
        # Agent position (assumes the agent is always on row 0)
        [[x]] = np.argwhere(grid[0, :] == 2)
        agent_col = x

        if (agent_col == 1 and action == Action.LEFT):
            return False
        elif (agent_col == 8 and action == Action.RIGHT):
            return False
        else:
            return True

    def feature_7(self, next_state, action):
        '''
        1 - if action a would likely take the agent into the safe area
        0 - otherwise
        '''
        cars = next_state[1]
        agent_road = cars['self']
        component_road = cars['others']
        component_num = len(component_road)
        safe_area_left_up = [agent_road[0]-agent_road[2], agent_road[1]]
        safe_area_left_down = [agent_road[0]-agent_road[2], agent_road[1]-agent_road[3]]
        safe_area_right_up = [agent_road[0]+agent_road[2], agent_road[1]]
        safe_area_right_down = [agent_road[0]+agent_road[2], agent_road[1]-agent_road[3]]
        agent_x = agent_road[0]
        agent_y = agent_road[1]
        if component_num == 0:
            return True
        else:
            if action == Action.LEFT:
                for component in range(component_num):
                    component_x = component_road[component][0]
                    component_y = component_road[component][1]
                    if component_x < (agent_x-agent_road[2]): 
                        component_x += component_road[component][2]
                        component_y -= component_road[component][3]
                        if not (component_x>safe_area_left_up[0]-agent_road[2] and component_y<safe_area_left_up[1]):
                            return True
                        else:
                            return False
                    else:
                        component_y -= component_road[component][3]
                        if not (component_x>safe_area_right_up[0]-agent_road[2] and component_y<safe_area_right_up[1]):
                            return True
                        else:
                            return False
            elif action == Action.RIGHT:
                for component in range(component_num):
                    component_x = component_road[component][0]
                    component_y = component_road[component][1]
                    if component_x < (agent_x+agent_road[2]):
                        component_x += component_road[component][2]
                        component_y -= component_road[component][3]
                        if not (component_x>safe_area_left_up[0]+agent_road[2] and component_y<safe_area_left_up[1]):
                            return True
                        else:
                            return False
                    else:
                        component_y -= component_road[component][3]
                        if not (component_x>safe_area_right_up[0]+agent_road[2] and component_y<safe_area_right_up[1]):
                            return True
                        else:
                            return False
            elif action == Action.ACCELERATE:
                for component in range(component_num):
                    component_x = component_road[component][0]
                    component_y = component_road[component][1]
                    if component_x < agent_x: 
                        component_x += component_road[component][2]
                        component_y -= component_road[component][3]
                        if not (component_x>safe_area_left_up[0] and component_y<safe_area_left_up[1]):
                            return True
                        else:
                            return False
                    else:
                        component_y -= component_road[component][3]
                        if not (component_x>safe_area_right_up[0] and component_y<safe_area_right_up[1]):
                            return True
                        else:
                            return False
            elif action == Action.BRAKE:
                for component in range(component_num):
                    component_x = component_road[component][0]
                    component_y = component_road[component][1]
                    if component_x < agent_x: 
                        component_x += component_road[component][2]
                        component_y -= component_road[component][3]
                        if not (component_x>safe_area_left_up[0] and component_y<safe_area_left_up[1]):
                            return True
                        else:
                            return False
                    else:
                        component_y -= component_road[component][3]
                        if not (component_x>safe_area_right_up[0] and component_y<safe_area_right_up[1]):
                            return True
                        else:
                            return False
            else:
                return False


    def feature_8(self, next_state):
        '''
        return the normalized speed
        '''
        speed = next_state[2]
        speed = speed / float(50)
        return speed

        

    def feature_9(self, next_state, action):
        '''
        1 - if the agent is not in the center of the road and the agent take the correct action to move it to the center
        0 - otherwise
        '''
        grid = next_state[3]
        # Agent position (assumes the agent is always on row 0)
        [[x]] = np.argwhere(grid[0, :] == 2)
        agent_col = x

        if agent_col < 4 and action == Action.RIGHT:
            return True
        elif agent_col > 5 and action == Action.LEFT:
            return True
        else:
            return False

    def feature_10(self, next_state,action):
        '''
        1 - if the speed < 0 and take accelerate action
        0 - otherwise
        '''
        speed = next_state[2]
        if speed < 0 and action == Action.ACCELERATE:
            return True
        else:
            return False

    def buildFeatures(self, state, next_state, action):
        '''
         Args:
            state[0] = road  -- 2-dimensional array containing [x, y] points
                     in pixel coordinates of the road grid
            state[1] = cars  -- dictionary which contains the location and the size
                     of the agent and the opponents in pixel coordinates
            state[2] = speed -- the relative speed of the agent with respect the others
            state[3] = grid  -- 2-dimensional numpy array containing the latest grid
                     representation of the environment
        '''

        predict_features = np.zeros(FEATURE_NUM)
        predict_features[0] = self.feature_1(next_state, action)
        predict_features[1] = self.feature_2(next_state, action)
        predict_features[2] = self.feature_3(state, next_state)
        predict_features[3] = self.feature_4(next_state, action)
        predict_features[4] = self.feature_5(next_state, action)
        predict_features[5] = self.feature_6(next_state, action)
        predict_features[6] = self.feature_7(next_state, action)
        predict_features[7] = self.feature_8(next_state)
        predict_features[8] = self.feature_9(next_state, action)
        predict_features[9] = self.feature_10(next_state, action)

        return predict_features


    def callback(self, learn, episode, iteration):
        """ Called at the end of each timestep for reporting/debugging purposes.
        """
        print "{0}/{1}: {2}".format(episode, iteration, self.total_reward)
        reward_List[self.count] = self.total_reward
        weight_List[self.count] = self.weights
        # You could comment this out in order to  speed up iterations
        cv2.imshow("Enduro", self._image)
        cv2.waitKey(1)


if __name__ == "__main__":
    a = FunctionApproximationAgent()
    a.run(True, episodes=500, draw=True)
    print 'Total reward: ' + str(a.total_reward)
    mean = np.mean(reward_List)
    variance = np.var(reward_List)
    print('mean:',mean)
    print('variance:',variance)


    x = np.arange(500)
    fig = plt.figure(0)
    ax1 = fig.add_subplot(211)
    ax1.set_ylabel('Weight Value')
    ax1.set_title('Weight 1 Value of Each Episode')
    ax1.plot(x,weight_List[:,0:1])
    ax2 = fig.add_subplot(212)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Weight Value')
    ax2.set_title('Weight 2 Value of Each Episode')
    ax2.plot(x, weight_List[:,1:2])

    fig = plt.figure(1)
    ax3 = fig.add_subplot(211)
    ax3.set_ylabel('Weight Value')
    ax3.set_title('Weight 3 Value of Each Episode')
    ax3.plot(x,weight_List[:,2:3])
    ax4 = fig.add_subplot(212)
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Weight Value')
    ax4.set_title('Weight 4 Value of Each Episode')
    ax4.plot(x, weight_List[:,3:4])

    fig = plt.figure(2)
    ax1 = fig.add_subplot(211)
    ax1.set_ylabel('Weight Value')
    ax1.set_title('Weight 5 Value of Each Episode')
    ax1.plot(x,weight_List[:,4:5])
    ax2 = fig.add_subplot(212)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Weight Value')
    ax2.set_title('Weight 6 Value of Each Episode')
    ax2.plot(x, weight_List[:,5:6])

    fig = plt.figure(3)
    ax3 = fig.add_subplot(211)
    ax3.set_ylabel('Weight Value')
    ax3.set_title('Weight 7 Value of Each Episode')
    ax3.plot(x,weight_List[:,6:7])
    ax4 = fig.add_subplot(212)
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Weight Value')
    ax4.set_title('Weight 8 Value of Each Episode')
    ax4.plot(x, weight_List[:,7:8])

    fig = plt.figure(4)
    ax1 = fig.add_subplot(211)
    ax1.set_ylabel('Weight Value')
    ax1.set_title('Weight 9 Value of Each Episode')
    ax1.plot(x,weight_List[:,8:9])
    ax2 = fig.add_subplot(212)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Weight Value')
    ax2.set_title('Weight 10 Value of Each Episode')
    ax2.plot(x, weight_List[:,9:10])

    plt.figure(5)
    plt.plot(x,reward_List)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward of Each Episode')
    plt.show()