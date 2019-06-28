function centroid = kmeans_centroid(mem, X, k, cur_center,shift)
    %Computes centroid
    a = [];
    [n,m,d] = size(X);
    ai = 1;
    for i=1:length(mem)
        if mem(i) == k
            opt_a = X(i,:,:);
            if d == 1
                a(ai,:) = opt_a;
            else
                a(ai,:,:) = opt_a;
            end
            ai = ai + 1;
        end
    end

    if size(a,1) == 0
        centroid = zeros(m, d); 
        return;
    end

    for d_i = 1:d
        centroid(:,d_i) = mean(a(:,:,d_i),1);
    end

end