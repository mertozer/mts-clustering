function [mem, cent, finalNorm, sqe] = multidim_kMeans(X, K, shift, cent_init)

    n = size(X, 1);
    m = size(X, 2);
    d = size(X, 3);
    Dist = zeros(n,K);
    if nargin < 4
        mem = ceil(K*rand(n, 1));
        cent = zeros(K, m, d);
    else
        cent = cent_init;
        for i = 1:n
            for k = 1:K
                Dist(i,k) = ED_dep_multi(...
                                    reshape(cent(k,:,:),m,d),...
                                    reshape(X(i,:,:),m,d));
            end
        end
        [~, mem] = min(Dist,[],2);
    end

    prevErr = -1;
    try_ = 0;

    for iter = 1:100
		tic;
		disp(iter);
        prev_mem = mem;

        for k = 1:K
            cent(k,:,:) = kmeans_centroid(mem, X, k, reshape(cent(k,:,:),m,d));     
        end

        for i = 1:n
            for k = 1:K
                Dist(i,k) = ED_dep_multi(reshape(X(i,:,:),m,d),reshape(cent(k,:,:),m,d));
            end
        end

        [val mem] = min(Dist,[],2);
        err_ = norm(prev_mem-mem);
        if err_ == 0
            break;
        else
            if err_ == prevErr
                try_ = try_ + 1;
                if try_ > 2
                    break
                end
            else
                prevErr = err_;
                try_ = 0;
            end
        end
        
		toc;
		disp(strcat('||PrevMem-CurMem||=',num2str(err_)));
    end
    finalNorm = norm(prev_mem-mem);
    sqe = 0;
    for i = 1:n
        sqe = sqe + Dist(i,mem(i));
    end
end


