# GridWorld implementation with simulations and estimations

## Running the code

To run the code with num_episodes and different policies:
`make run episodes=num_episodes policy=required_policy condition=False`

So to run for 10000 episodes and the random policy
`make run episodes=10000 policy=uniform`

Policy can take the following values:
`policy=uniform`
`policy=optimal1`
`policy=optimal2`
`policy=goRight`

Condition is True when you want to calculate the conditional probability in question and False otherwise

To clean the current directory run:
`make clean_all`


