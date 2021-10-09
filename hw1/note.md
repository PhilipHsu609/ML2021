# PLA
## Requirements
- N = 100 examples
- 10 features
- Training set
  - First 10 numbers of the line is $\textbf{x}_n$, the last number if $y_n$
- Initialize $\textbf{w} = 0$
- sign(0) = -1
- Add $x_0$ = 1 to every $\textbf{x}_n$.
- Randomly picks an example in every iteration and updates $\textbf{w}_t$ iff $\textbf{w}_t$ is incorrect on this example.
  - With replacement: same example can be picked multiple times.
- If $\textbf{w}_t$ is correct consecutively after checking 5N randomly-picked examples, PLA halt.

## Question 13
- Repeat 1000 times, each with different random seed.
- Average squared length of $\textbf{w}_{PLA}$?

## Question 14
- Scale up each $\textbf{x}_n$ by 2, including scaling $x_0$ from 1 to 2.
- Repeat 1000 times.
- Average squared length of $\textbf{w}_{PLA}$?

## Question 15
- Scale down each $\textbf{x}_n$ including $x_0$ by $||\textbf{x}_n||$.
- Repeat 1000 times.
- Average squared length of $\textbf{w}_{PLA}$?

## Question 16
- Set $x_0$ to 0
- Repeat 1000 times.
- Average squared length of $\textbf{w}_{PLA}$?