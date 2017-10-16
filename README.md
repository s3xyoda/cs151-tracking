# cs151-tracking

## Project 5: Ghostbusters

<div class="project">

![GHOSTBUSTERS](https://github.com/HEATlab/cs151-tracking/blob/master/busters.png)

> <center>I can hear you, ghost.
> Running won't save you from my
> Particle filter!</center>

### <a name="Introduction"></a>Introduction

Pacman spends his life running from ghosts, but things were not always so. Legend has it that many years ago, Pacman's great grandfather Grandpac learned to hunt ghosts for sport. However, he was blinded by his power and could only track ghosts by their banging and clanging.

In this project, you will design Pacman agents that use sensors to locate and eat invisible ghosts. You'll advance from locating single, stationary ghosts to hunting packs of multiple moving ghosts with ruthless efficiency.

The code for this project contains the following files, available as a [zip archive](https://github.com/HEATlab/cs151-tracking/archive/master.zip).

<table class="intro" border="0" cellpadding="10">

<tbody>

<tr>

<td colspan="2">**Files you'll edit:**</td>

</tr>

<tr>

<td>bustersAgents.py</td>

<td>Agents for playing the Ghostbusters variant of Pacman.</td>

</tr>

<tr>

<td>inference.py</td>

<td>Code for tracking ghosts over time using their sounds.</td>

</tr>

<tr>

<td colspan="2">**Files you will not edit:**</td>

</tr>

<tr>

<td>busters.py</td>

<td>The main entry to Ghostbusters (replacing Pacman.py)</td>

</tr>

<tr>

<td>bustersGhostAgents.py</td>

<td>New ghost agents for Ghostbusters</td>

</tr>

<tr>

<td>distanceCalculator.py</td>

<td>Computes maze distances</td>

</tr>

<tr>

<td>game.py</td>

<td>Inner workings and helper classes for Pacman</td>

</tr>

<tr>

<td>ghostAgents.py</td>

<td>Agents to control ghosts</td>

</tr>

<tr>

<td>graphicsDisplay.py</td>

<td>Graphics for Pacman</td>

</tr>

<tr>

<td>graphicsUtils.py</td>

<td>Support for Pacman graphics</td>

</tr>

<tr>

<td>keyboardAgents.py</td>

<td>Keyboard interfaces to control Pacman</td>

</tr>

<tr>

<td>layout.py</td>

<td>Code for reading layout files and storing their contents</td>

</tr>

<tr>

<td>util.py</td>

<td>Utility functions</td>

</tr>

</tbody>

</table>

**Files to Edit and Submit:** You will fill in portions of [bustersAgents.py](https://github.com/HEATlab/cs151-tracking/blob/master/bustersAgents.py) and [inference.py](https://github.com/HEATlab/cs151-tracking/blob/master/inference.py) during the assignment. You should submit these files with your code and comments. Please _do not_ change the other files in this distribution or submit any of our original files other than these files.

**Evaluation:** Your code will be autograded for technical correctness. Please _do not_ change the names of any provided functions or classes within the code, or you will wreak havoc on the autograder. However, the correctness of your implementation -- not the autograder's judgements -- will be the final judge of your score. If necessary, we will review and grade assignments individually to ensure that you receive due credit for your work.

**Academic Dishonesty:** We will be checking your code against other submissions in the class for logical redundancy. If you copy someone else's code and submit it with minor changes, we will know. These cheat detectors are quite hard to fool, so please don't try. We trust you all to submit your own work only; _please_ don't let us down. If you do, we will pursue the strongest consequences available to us.

**Getting Help:** You are not alone! If you find yourself stuck on something, contact the course staff for help. Office hours, section, and the discussion forum are there for your support; please use them. If you can't make our office hours, let us know and we will schedule more. We want these projects to be rewarding and instructional, not frustrating and demoralizing. But, we don't know when or how to help unless you ask.

**Discussion:** Please be careful not to post spoilers.

</div>

<div class="project">

### <a name="Welcome"></a>Ghostbusters and BNs

In the cs151 version of Ghostbusters, the goal is to hunt down scared but invisible ghosts. Pacman, ever resourceful, is equipped with sonar (ears) that provides noisy readings of the Manhattan distance to each ghost. The game ends when Pacman has eaten all the ghosts. To start, try playing a game yourself using the keyboard.

<pre>python busters.py</pre>

The blocks of color indicate where the each ghost could possibly be, given the noisy distance readings provided to Pacman. The noisy distances at the bottom of the display are always non-negative, and always within 7 of the true distance. The probability of a distance reading decreases exponentially with its difference from the true distance.

Your primary task in this project is to implement inference to track the ghosts. For the keyboard based game above, a crude form of inference was implemented for you by default: all squares in which a ghost could possibly be are shaded by the color of the ghost. Naturally, we want a better estimate of the ghost's position. Fortunately, Bayes' Nets provide us with powerful tools for making the most of the information we have. Throughout the rest of this project, you will implement algorithms for performing both exact and approximate inference using Bayes' Nets. The project is challenging, so we do encourage you to start early and seek help when necessary.

While watching and debugging your code with the autograder, it will be helpful to have some understanding of what the autograder is doing. There are 2 types of tests in this project, as differentiated by their `*.test` files found in the subdirectories of the `test_cases` folder. For tests of class `DoubleInferenceAgentTest`, your will see visualizations of the inference distributions generated by your code, but all Pacman actions will be preselected according to the actions of the staff implementation. This is necessary in order to allow comparision of your distributions with the staff's distributions. The second type of test is `GameScoreTest`, in which your `BustersAgent` will actually select actions for Pacman and you will watch your Pacman play and win games.

As you implement and debug your code, you may find it useful to run a single test at a time. In order to do this you will need to use the -t flag with the autograder. For example if you only want to run the first test of question 1, use:

<pre>python autograder.py -t test_cases/q1/1-ObsProb</pre>

In general, all test cases can be found inside test_cases/q*.

</div>

### QUESTION 0 (0 POINTS): `DiscreteDistribution` Class

Throughout this project, we will be using the `DiscreteDistribution` class defined in `inference.py` to model belief distributions and weight distributions. This class is an extension of the built-in Python dictionary class, where the keys are the different discrete elements of our distribution, and the corresponding values are proportional to the belief or weight that the distribution assigns that element. This question asks you to fill in the missing parts of this class, which will be crucial for later questions (even though this question itself is worth no points).

First, fill in the `normalize` method, which normalizes the values in the distribution to sum to one, but keeps the proportions of the values the same. Use the `total` method to find the sum of the values in the distribution. For an empty distribution or a distribution where all of the values are zero, do nothing. Note that this method modifies the distribution directly, rather than returning a new distribution.

Second, fill in the `sample` method, which draws a sample from the distribution, where the probability that a key is sampled is proportional to its corresponding value. Assume that the distribution is not empty, and not all of the values are zero. Note that the distribution does not necessarily have to be normalized prior to calling this method. You may find Python's built-in `random.random()` function useful for this question.

There are no autograder tests for this question, but the correctness of your implementation can be easily checked. We have provided [Python doctests](https://docs.python.org/2/library/doctest.html) as a starting point, and you can feel free to add more and implement other tests of your own. You can run the doctests using:

<pre>python -m doctest -v inference.py</pre>

Note that, depending on the implementation details of the `sample` method, some correct implementations may not pass the doctests that are provided. To thoroughly check the correctness of your `sample` method, you should instead draw many samples and see if the frequency of each key converges to be proportional of its corresponding value.

### QUESTION 1 (2 POINTS): Observation Probability

In this question, you will implement the `getObservationProb` method in the `InferenceModule` base class in `inference.py`. This method takes in an observation (which is a noisy reading of the distance to the ghost), Pacman's position, the ghost's position, and the position of the ghost's jail, and returns the probability of the noisy distance reading given Pacman's position and the ghost's position. In other words, we want to return `P(noisyDistance | pacmanPosition, ghostPosition)`.

The distance sensor has a probability distribution over distance readings given the true distance from Pacman to the ghost. This distribution is modeled by the function `busters.getObservationProbability(noisyDistance, trueDistance)`, which returns `P(noisyDistance | trueDistance)` and is provided for you. You should use this function to help you solve the problem, and use the provided `manhattanDistance` function to find the distance between Pacman's location and the ghost's location.

However, there is the special case of jail that we have to handle as well. Specifically, when we capture a ghost and send it to the jail location, our distance sensor deterministically returns `None`, and nothing else. So, if the ghost's position is the jail position, then the observation is `None` with probability 1, and everything else with probability 0\. Conversely, if the distance reading is not `None`, then the ghost is in jail with probability 0\. If the distance reading is None, then the ghost is in jail with probability 1\. Make sure you handle this special case in your implementation.

To test your code and run the autograder for this question:

<pre>python autograder.py -q q1</pre>

As a general note, it is possible for some of the autograder tests to take a long time to run for this project, and you will have to exercise patience. As long as the autograder doesn't time out, you should be fine (provided that you actually pass the tests).

### QUESTION 2 (3 POINTS): Exact Inference Observation

In this question, you will implement the `observeUpdate` method in `ExactInference` class of `inference.py` to correctly update the agent's belief distribution over ghost positions given an observation from Pacman's sensors. <span style="font-family: 'Open Sans', 'Helvetica Neue', Helvetica, Arial, sans-serif; line-height: 22.4px;">You are implementing the online belief update for observing new evidence.</span> The observe method should,<span style="line-height: 25.6px;"> for this problem, update the belief at every position on the map after receiving a sensor reading. You should iterate your updates over the variable self.allPositions which includes all legal positions plus the special jail position. Beliefs represent the probability that the ghost is at a particular location, and are stored as a </span><span style="font-family: monospace, serif; line-height: 22.4px;">DiscreteDistribution</span><span style="line-height: 25.6px; font-size: 1em;"> object in a field called </span><span style="font-family: monospace, serif; line-height: 22.4px;">self.beliefs</span><span style="line-height: 25.6px; font-size: 1em;">, which you should update.</span>

<span style="line-height: 25.6px;">Before typing any code, </span><span style="line-height: 25.6px;">write down the equation of the inference problem you are trying to solve. </span>You should use the function <span style="font-family: monospace, serif; line-height: 22.4px;">self.getObservationProb</span><span style="line-height: 25.6px;"> that you wrote in the last question, which returns the probability of an observation given Pacman's position, a potential ghost position, and the jail position. You can obtain Pacman's position using </span><span style="font-family: monospace, serif; line-height: 22.4px;">gameState.getPacmanPosition()</span><span style="font-size: 1em; line-height: 25.6px;">, and the jail position using </span><span style="font-family: monospace, serif; line-height: 22.4px;">self.getJailPosition()</span><span style="font-size: 1em; line-height: 25.6px;">.</span>

<span style="line-height: 1.6;">In the Pacman display, high posterior beliefs are represented by bright colors, while low beliefs are represented by dim colors. You should start with a large cloud of belief that shrinks over time as more evidence accumulates. </span>As you watch the test cases, be sure that you understand how the squares converge to their final coloring. 

_Note:_ your busters agents have a separate inference module for each ghost they are tracking. That's why if you print an observation inside the `update` function, you'll only see a single number even though there may be multiple ghosts on the board.

To run the autograder for this question and visualize the output:

<pre style="font-size: 16px; line-height: 25.6px;">python autograder.py -q q2</pre>

If you want to run this test (or any of the other tests) without graphics you can add the following flag:

<pre style="text-rendering: optimizeLegibility; font-size: 16px; padding: 0px; border: 0px; outline: 0px; font-stretch: inherit; line-height: 1.4em; vertical-align: baseline;">python autograder.py -q q2 --no-graphics</pre>

***IMPORTANT***: In general, it is possible sometimes for the autograder to time out if running the tests with graphics. To accurately determine whether or not your code is efficient enough, you should run the tests with the <span style="font-family: monospace, serif; line-height: 1.4em; white-space: pre-wrap;">--no-graphics</span><span style="font-size: 1em; line-height: 1.6em;"> flag. If the autograder passes with this flag, then you</span><span style="font-size: 1em; line-height: 1.6em;"> will receive full points, even if the autograder times out</span><span style="font-size: 1em; line-height: 1.6em;"> with </span><span style="font-size: 1em; line-height: 1.6em;">graphics.</span>

### QUESTION 3 (3 POINTS): Exact Inference with Time Elapse

In the previous question you implemented belief updates for Pacman based on his observations. Fortunately, Pacman's observations are not his only source of knowledge about where a ghost may be. Pacman also has knowledge about the ways that a ghost may move; namely that the ghost can not move through a wall or more than one space in one time step.

<span style="font-family: 'Open Sans', 'Helvetica Neue', Helvetica, Arial, sans-serif; line-height: 25.6px;">To understand why this is useful to Pacman, consider the following scenario in which there is Pacman and one Ghost. Pacman receives many observations which indicate the ghost is very near, but then one which indicates the ghost is very far. The reading indicating the ghost is very far is likely to be the result of a buggy sensor. Pacman's prior knowledge of how the ghost may move will decrease the impact of this reading since Pacman knows the ghost could not move so far in only one move.</span>

In this question, you will implement the `elapseTime` method in `ExactInference`<span style="line-height: 25.6px;">. </span><span style="line-height: 25.6px; font-family: 'Open Sans', 'Helvetica Neue', Helvetica, Arial, sans-serif;">The elapseTime step should,</span><span style="line-height: 25.6px; text-rendering: optimizeLegibility; margin: 0px; padding: 0px; border: 0px; outline: 0px; font-stretch: inherit; font-family: 'Open Sans', 'Helvetica Neue', Helvetica, Arial, sans-serif; vertical-align: baseline;"> for this problem, update the belief at every position on the map after one time step elapsing. </span><span style="font-size: 1em; line-height: 1.6em;">Your agent has access to the action distribution for the ghost through </span>`self.getPositionDistribution`<span style="font-size: 1em; line-height: 1.6em;">. In order to obtain the distribution over new positions for the ghost, given its previous position, use this line of code:</span>

<pre style="font-size: 16px; line-height: 25.6px;">newPosDist = self.getPositionDistribution(gameState, oldPos)</pre>

Where `oldPos` refers to the previous ghost position. `newPosDist` is a `DiscreteDistribution` object, where for each position `p` in `self.allPositions`, `newPosDist[p]` is the probability that the ghost is at position `p` at time `t + 1`, given that the ghost is at position `oldPos` at time `t`. Note that this call can be fairly expensive, so if your code is timing out, one thing to think about is whether or not you can reduce the number of calls to `self.getPositionDistribution`.

<span style="line-height: 25.6px;">Before typing any code, </span><span style="line-height: 25.6px;">write down the equation of the inference problem you are trying to solve. </span>In order to test your `predict` implementation separately from your `update` implementation in the previous question, this question will not make use of your `update` implementation.

Since Pacman is not observing the ghost, this means the ghost's actions will not impact Pacman's beliefs. Over time, Pacman's beliefs will come to reflect places on the board where he believes ghosts are most likely to be given the geometry of the board and what Pacman already knows about their valid movements.

For the tests in this question we will sometimes use a ghost with random movements and other times we will use the GoSouthGhost. This ghost tends to move south so over time, and without any observations, Pacman's belief distribution should begin to focus around the bottom of the board. To see which ghost is used for each test case you can look in the .test files.

To run the autograder for this question and visualize the output:

<pre style="font-size: 16px; line-height: 25.6px;">python autograder.py -q q3</pre>

If you want to run this test (or any of the other tests) without graphics you can add the following flag:

<pre style="text-rendering: optimizeLegibility; font-size: 16px; padding: 0px; border: 0px; outline: 0px; font-stretch: inherit; line-height: 1.4em; vertical-align: baseline;">python autograder.py -q q3 --no-graphics</pre>

***IMPORTANT***<span style="line-height: 25.6px;">: In general, it is possible sometimes for the autograder to time out if running the tests with graphics. To accurately determine whether or not your code is efficient enough, you should run the tests with the </span><span style="font-family: monospace, serif; line-height: 1.4em; white-space: pre-wrap;">--no-graphics</span><span style="font-size: 1em; line-height: 1.6em;"> flag. If the autograder passes with this flag, then you</span><span style="font-size: 1em; line-height: 1.6em;"> will receive full points, even if the autograder times out</span><span style="font-size: 1em; line-height: 1.6em;"> with </span><span style="font-size: 1em; line-height: 1.6em;">graphics.</span>

As you watch the autograder output, remember that lighter squares indicate that pacman believes a ghost is more likely to occupy that location, and darker squares indicate a ghost is less likely to occupy that location. For which of the test cases do you notice differences emerging in the shading of the squares? Can you explain why some squares get lighter and some squares get darker?

### QUESTION 4 (2 POINTS): Exact Inference Full Test

Now that Pacman knows how to use both his prior knowledge and his observations when figuring out where a ghost is, he is ready to hunt down ghosts on his own. This question will use your `observeUpdate` and `elapseTime` implementations together, along with a simple greedy hunting strategy which you will implement for this question. In the simple greedy strategy, Pacman assumes that each ghost is in its most likely position according to his beliefs, then moves toward the closest ghost. Up to this point, Pacman has moved by randomly selecting a valid action.

Implement the `chooseAction` method in `GreedyBustersAgent` in `bustersAgents.py`. Your agent should first find the most likely position of each remaining uncaptured ghost, then choose an action that minimizes the maze distance to the closest ghost.

To find the maze distance between any two positions <span style="font-family: monospace, serif; line-height: 25.6px;">pos1</span><span style="font-size: 1em; line-height: 1.6em;"> and </span><span style="font-family: monospace, serif; line-height: 25.6px;">pos2</span><span style="font-size: 1em; line-height: 1.6em;">, use </span>`self.distancer.getDistance(pos1, pos2)`<span style="font-size: 1em; line-height: 1.6em;">. To find the successor position of a position after an action:</span>

<pre style="font-size: 16px; line-height: 25.6px;">successorPosition = Actions.getSuccessor(position, action)</pre>

You are provided with `livingGhostPositionDistributions`, a list of `DiscreteDistribution` objects representing the position belief distributions for each of the ghosts that are still uncaptured.

If correctly implemented, your agent should win the game in `q3/3-gameScoreTest` with a score greater than 700 at least 8 out of 10 times. _Note:_ the autograder will also check the correctness of your inference directly, but the outcome of games is a reasonable sanity check.

To run the autograder for this question and visualize the output:

<pre style="font-size: 16px; line-height: 25.6px;">python autograder.py -q q4</pre>

If you want to run this test (or any of the other tests) without graphics you can add the following flag:

<pre style="font-size: 16px; line-height: 25.6px;">python autograder.py -q q4 --no-graphics</pre>

***IMPORTANT***<span style="line-height: 25.6px;">: In general, it is possible sometimes for the autograder to time out if running the tests with graphics. To accurately determine whether or not your code is efficient enough, you should run the tests with the </span><span style="font-family: monospace, serif; line-height: 1.4em; white-space: pre-wrap;">--no-graphics</span><span style="font-size: 1em; line-height: 1.6em;"> flag. If the autograder passes with this flag, then you</span><span style="font-size: 1em; line-height: 1.6em;"> will receive full points, even if the autograder times out</span><span style="font-size: 1em; line-height: 1.6em;"> with </span><span style="font-size: 1em; line-height: 1.6em;">graphics.</span>

<span style="font-size: 1em; line-height: 1.6em;"> </span>


