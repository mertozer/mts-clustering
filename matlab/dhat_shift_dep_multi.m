function [dist, optshift, opty] = dhat_shift_dep_multi(x, y, W)
    if size(x,1) < size(x,2)
        x = x';
    end
    if size(y,1) < size(y,2)
        y = y';
    end
    
    d = size(x,2);
    if d ~= size(y,2)
        display('This should not happen: multi_dhat_shift');
    end
    min_d = 0;
    for d_i = 1:d
        min_d = min_d + scale_d(x(:,d_i)',y(:,d_i)');
    end
    
    range = -W:W;
    opty = y;
    optshift = 0;
    for shift = range
        if shift < 0
            yshift = [y(-shift + 1:end,:); zeros(-shift,d)];
        else
            yshift = [zeros(shift,d); y(1:end-shift,:) ];
        end
        cur_d = 0;
        for d_i = 1:d
            cur_d = cur_d + scale_d(x(:,d_i)',yshift(:,d_i)');
        end
        if cur_d <= min_d
            optshift = shift;
            opty = yshift;
            min_d = cur_d;
        end
    end
    optshift = ones(1,d)*shift;
    dist = min_d;
end

function dist = scale_d(x,y)
    alpha = x * y' / ((y * y')+eps);
    dist = norm(x - alpha * y) / (norm(x) + eps);
end