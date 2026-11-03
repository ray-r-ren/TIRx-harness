I need an RMSNorm kernel for our H100. Input `x` is a float32 matrix of 16 rows × 512 columns
(row-major) and `w` is a float32 weight vector of length 512:

    out[r, c] = x[r, c] / sqrt(mean(x[r, :]²) + 1e-5) * w[c]

The kernel's parameters must be exactly `x`, `w`, `out` (in that order). Give me one self-contained
kernel `.py` file, including the test inputs and the NumPy reference you checked it against, so I
can rerun your checks. It must compute the right numbers and be free of races and deadlocks. In
your answer, say how you verified it.

---
Your files are in `/task/inputs`. Save every file you produce in `/task/outputs`, and write your final answer to the user in `/task/answer.txt`.
