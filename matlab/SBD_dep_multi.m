function [dist, optshift, yshift]= SBD_dep_multi(x, y, maxshift)
    [m,d] = size(x);
    
    cc_ = 0;
    for d_i = 1:d
        cc_d_i = NCCc(x(:,d_i),y(:,d_i));
        cc_ = cc_ + cc_d_i;
    end
    cc = zeros(size(cc_));
    cc(length(x)) = cc_(length(x));
    for i = 1:maxshift
        cc(length(x)+i) = cc_(length(x)+i);
        cc(length(x)-i) = cc_(length(x)-i);
    end
    [maxCC,maxCCI]=max(cc);

    shift = maxCCI - max(length(x),length(y));

    if shift < 0
            yshift = [y(-shift + 1:end,:); zeros(-shift,d)];
        else
            yshift = [zeros(shift,d); y(1:end-shift,:) ];
    end
    optshift = ones(1,d)*shift;
    dist = d - maxCC;
end