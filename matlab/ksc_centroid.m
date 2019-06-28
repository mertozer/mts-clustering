function ksc = ksc_centroid(mem, X, k, cur_center, shift)
    %Computes ksc centroid
    
    [n,m,d] = size(X);
    if d == 1
        a = zeros(length(find(mem==k)),m);
    else
        a = zeros(length(find(mem==k)),m,d);
    end
    ai = 1;
    sum_cur_center = sum(cur_center(:));
    for i=1:length(mem)
        if mem(i) == k
            if sum_cur_center == 0
                opt_a = X(i,:,:);
            else
                [~, ~, opt_a] = dhat_shift_dep_multi(reshape(cur_center,m,d), ...
                                            reshape(X(i,:,:),m,d),shift);
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
        ksc = zeros(m, d); 
        return;
    end
    for d_i = 1:d
        a_di = a(:,:,d_i);
        b = a_di ./ repmat(sqrt(sum(a_di.^2,2))+eps, [1 m]);
        M = b'*b - n * eye(m);
        [V, D] = eig(M);

        ksc_di = V(:,end);

        finddistance1 = sqrt(sum((a_di(1,:) - ksc_di').^2));
        finddistance2 = sqrt(sum((a_di(1,:) - (-ksc_di')).^2));

        if (finddistance1<finddistance2)
            ksc_di = ksc_di;
        else
            ksc_di = -ksc_di;
        end

        if sum(ksc_di) < 0
            ksc_di = -ksc_di;
        end
        ksc(:,d_i) = ksc_di;
    end
end
