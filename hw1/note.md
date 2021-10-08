# PLA
## Requirements
- N = 100 examples
- 10 features
- Training set
  - First 10 numbers of the line is x_n, the last number if y_n
- Initialize w = 0
- sign(0) = -1
- Add x_0 = 1 to every x_n.
- Randomly picks an example in every iteration and updates w_t iff w_t is incorrect on this example.
  - with replacement: same example can be picked multiple times
- If w_t is correct consecutively after checking 5N randomly-picked examples, PLA halt.

## Question 13
- Repeat 1000 times, each with different random seed.
- Average squared length of w_PLA?

## Question 14
- Scale up each x_n by 2, including scaling x_0 from 1 to 2.
- Repeat 1000 times.
- Average squared length of w_PLA?

## Question 15
- Scale down each x_n including x_0 by ||x_n||.
- Repeat 1000 times.
- Average squared length of w_PLA?

## Question 16
- Set x_0 to 0
- Repeat 1000 times.
- Average squared length of w_PLA?