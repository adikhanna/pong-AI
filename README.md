 In 1972, the game of Pong was released by Atari. Despite its simplicity, it was a ground-breaking game in its day, and it has been credited as having helped launch the video game industry. In this project, I attempted to create a simple version of Pong and use Q-learning (TD-learning), SARSA and Deep learning to train agents to play the game.

The deep learning agent makes use of policy behavioral cloning, which means it trains itself using an expert policy. This in turn allows for continous state spaces to be enumerated as the neural net allows the agent to learna mapping between any game state and subsequent action. 

This deep learning agent makes use of a network of stacked affine transformations and non-linear activation functions. The neural net makes use of four layers, wherein each layer has its own weight and bias vectors. 
