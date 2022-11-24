#Convolutional Auto-Encoder

Let A,B be 2 tensors of the same size (n, m) where n is the number of rows, m the number of columns.
The MSELoss is defined as:

	MSELoss(A,B) = \frac{1}{N\times M} \sum{i,j}^{N,M}{(A_{ij}-B_{ij})^2}


