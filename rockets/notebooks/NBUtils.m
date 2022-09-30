classdef NBUtils
methods(Static)
function plotIt(params, tx, state)
    angx = atan2(state(2,:), state(1,:));
    angx = angx - params.earthThetaD * (tx - tx(1));
    rx = vecnorm(state(1:2,:));
    xx = rx .* cos(angx);
    yx = rx .* sin(angx);
    plot(xx, yx);
end
end
end