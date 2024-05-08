units = [16 16 2];
activations = {@softplus, @softplus, @linear};

fpath = fullfile('..', 'weights.txt');
[W, b] = read_weights(units,fpath);

x = [2; 2];
[y, dy, ddy] = analytical(x);

[ynn, dynn, ddynn] = mlp(W,b,x,activations);

