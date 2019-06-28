function [Dist, optshift, opty] = ED_dep_multi(x,y,shift)
    [m,d] = size(x);
    if d~= size(y,2)
        display('This should not happen: ED_multi');
    end
    Dist = 0;
    for m_i = 1:m
        for d_i = 1:d
            Dist = Dist + (x(m_i,d_i) - y(m_i,d_i))^2;
        end
    end
    Dist = sqrt(Dist);
    optshift = 0;
    opty = y;
end