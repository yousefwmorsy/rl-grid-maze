# Summary
> ### Repo Link: https://github.com/yousefwmorsy/rl-grid-maze
## Custom Grid Maze Environment

- Grid size: A configurable square grid, defaulting to 5×5.
- Observation space: A dictionary containing:
  - Agent position (row, column).
  - Goal position (row, column).
  - Two bad-cell positions (each represented by row and column).
- Action space: Four discrete actions:
  - 0: Right
  - 1: Up
  - 2: Left
  - 3: Down
- Movement: Deterministic movement. The agent moves one cell in the selected direction.
- Reward function:
  - +1 for reaching the goal.
  - -1 for reaching a bad cell.
  - 0 for other movements.
- Environment initialization: Randomly generates the agent's starting position, goal, and two distinct bad cells. Bad cells cannot overlap the start or goal.
- Rendering: Uses PyGame to visualize the maze, with different colors and labels for the starting cell, goal, bad cells, and agent.

## Policy Iteration Algorithm
### Initialization

- Receives the environment, discount factor, and convergence threshold.
- Initializes a random policy if no initial policy is provided.
- Initializes the value function to zeros if no initial values are provided.

### Policy Evaluation

Evaluates the current policy by repeatedly updating the value function until the maximum change between successive updates becomes smaller than the convergence threshold.

### Policy Improvement

Evaluates all possible actions for each state and selects the action with the highest action value. The algorithm checks whether any action has changed. If no actions change, the policy is considered stable.

### Policy Iteration

Alternates between policy evaluation and policy improvement until a stable policy is reached. The implementation also tracks the number of policy improvement iterations required for convergence.

### Results Visualization

Uses NumPy to reshape the policy and value function into grid representations and prints them using directional arrows and corresponding state values.

# Q&A
## What is the state-space size of the 5x5 Grid Maze problem?

5 rows x 5 columns = 25 states

All possible state configurations are 25^4 approx. (we may remove invalid maze configs -> 25x24x23x22 )

## How to optimize the policy iteration for the Grid Maze problem?

Add a maximum number of iterations. Use matrices and vectors instead of loops. 

## How many iterations did it take to converge on a stable policy for 5x5 maze?

It took 5-8 iterations on average (different seeds) to converge on a stable optimal policy.

## Explain, with an example, how policy iteration behaves with multiple goal cells.

With multiple goals, the algorithm evaluates the best path to any of them, we get a policy that leads to the nearest goal (as long as we have a discount factor).

## Can policy iteration work on a 100x100 maze? Explain why?

Yes but it will take a very long time as well as use up a lot of memory. 100x100 = 10000 states

## Can policy iteration work on a continuous-space maze? Explain why?

No, the policy iteration depends on having a table of states and actions, which is not feasible in continuous space.

## Can policy iteration work with moving bad cells (like Packman moving ghosts)? Explain why?

Yes, as long as it still follows MDP (next state depends only on the current state and action). However it adds complexity since ghost positions would be added to the state representation.