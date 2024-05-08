function [y,dy,ddy] = analytical(x)
%  Return the analytical function values and derivatives
%   Detailed explanation goes here
x2 = x.^2;
x3 = x.^3;
y = [x3(1)*x3(2);
     x3(1)*x3(2)];
if nargout > 1
    dy = [3*x2(1)*x3(2) 3*x3(1)*x2(2);
          3*x2(1)*x3(2) 3*x3(1)*x2(2)];
    if nargout > 2
        ddy = [6*x(1)*x3(2) 9*x2(1)*x2(2);
               6*x(1)*x3(2) 9*x2(1)*x2(2)
               9*x2(1)*x2(2) 6*x3(1)*x(2);
               9*x2(1)*x2(2) 6*x3(1)*x(2)];
    end
end
end

