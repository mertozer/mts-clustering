function centroid = kshape_centroid(mem, X, k, cur_center, shift)
    %Computes centroid
    a = [];
    [n,m,d] = size(X);
    ai = 1;
    for i=1:length(mem)
        if mem(i) == k
            if sum(cur_center(:)) == 0
                opt_a = X(i,:,:);
            else
                %align to previous center
                [tmp, tmps, opt_a] = SBD_dep_multi(reshape(cur_center,m,d), reshape(X(i,:,:),m,d), shift);
            end
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
        [~, ncolumns]=size(a(:,:,d_i));
        [Y, mean2, std2] = zscore(a(:,:,d_i),[],2);
        S = Y' * Y;
        P = (eye(ncolumns) - 1 / ncolumns * ones(ncolumns));
        M = P*S*P;
        if sum(sum(M)) == 0
            centroid(:,d_i) = zeros(1, size(X,2)); 
        end
        [V, D] = qdwheig(M);

        centroid_di = V(:,end);

        finddistance1 = sqrt(sum((a(1,:,d_i) - centroid_di').^2));
        finddistance2 = sqrt(sum((a(1,:,d_i) - (-centroid_di')).^2));

        if (finddistance1<finddistance2)
            centroid_di = centroid_di;
        else
            centroid_di = -centroid_di;
        end

        centroid_di = zscore(centroid_di);
        centroid(:,d_i) = centroid_di;
    end
end
