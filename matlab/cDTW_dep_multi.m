function [Dist, optshift , opty] = cDTW_dep_multi(x,y,W)
    
    [m,d]=size(x);
    if d ~= size(y,2)
        display('This should not happen cDTW');
    end
    
    Dist = 0;
    D=ones(m+1,m+1)*inf;

    D(1,1) = 0;
    for i=2:m+1
        for j=max(2, i-W):min(m+1, i+W)
            cost = 0; 
            for d_i = 1:d
                cost = cost + (x(i-1,d_i)-y(j-1,d_i))^2;
            end
            D(i,j)=sqrt(cost)+min([D(i-1,j),D(i-1,j-1),D(i,j-1)]);
        end
    end
    Dist = D(m+1, m+1);
    opty = y;
    optshift = zeros(1,d);
end