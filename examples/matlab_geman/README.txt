
This demo replicates a classic experiment due to Geman and Geman's seminal
work

   [Geman1984]  Stuart Geman, Donald Geman, "Stochastic relaxation, Gibbs
distributions, and the Bayesian restoration of images", IEEE Transactions on
Pattern Analysis and Machine Intelligence, Vol. 6, pages 721-741, 1984.

A single image is sampled from a known model, then noise is added, and a
reconstruction of the noise-free image is made using maximum aposteriori
inference.

The demo illustrates basic operations in grante: how to set up a factor graph,
obtain samples, and solve basic inference problems.

