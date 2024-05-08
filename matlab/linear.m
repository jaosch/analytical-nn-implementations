function [y,dy,ddy] = linear(x)
y = x;
if nargout > 1
    dy = x./x;
    if nargout > 2
        ddy = x * 0;
    end
end
end