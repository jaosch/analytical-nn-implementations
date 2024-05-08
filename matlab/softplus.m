function [y,dy,ddy] = softplus(x)
y = softplus(x);
if nargout > 1
    dy = sigmoid(x);
    if nargout > 2
        ddy = dsigmoid(dy);
    end
end

function y = softplus(x)
    y = log(1 + exp(x));
end
function y = sigmoid(x)
    y = 1 ./ (1 + exp(-x));
end
function y = dsigmoid(x)
    y = x.*(1 - x);
end
end
