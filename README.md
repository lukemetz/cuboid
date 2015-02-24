# cuboid
Contains a number of extensions to [Blocks](https://github.com/bartvm/blocks).

Currently contains:

**bricks.BatchNormalizationConv** and **bricks.BatchNormalization**, batch
normalization for conv and nonconv layers.

**bricks.FilterPool** Filter pool

**algorithms.AdaM** update algorithm.

**algorithms.NAG** Nesterov momentum update algorithm.

**extensions.LogToFile** logs all stats to csv's each epoch.

**extensions.EpochProgress** show a progress bar while in the middle of
an epoch

**datasets.DataStreamBackground** wrap a datastream and make it run in a
different thread. Great for slow execution.
