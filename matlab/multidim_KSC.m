function [mem, cent, finalNorm, sqe] = multidim_KSC(X, K, shift, cent_init)
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
                Dist(i,k) = dhat_shift_dep_multi(...
                                    reshape(cent(k,:,:),m,d),...
                                    reshape(X(i,:,:),m,d),...
                                    shift...
                                    );
            end
        end
        [~, mem] = min(Dist,[],2);
    end

    prevErr = -1;
    try_ = 0;
    
    for iter = 1:100
		disp(strcat('Iteration-',num2str(iter)));
		tic;
        prev_mem = mem;
        
        for k = 1:K
            cent(k,:,:) = ksc_centroid(mem, X, k, reshape(cent(k,:,:),m,d),shift);
        end
        for i = 1:n
            for k = 1:K
                Dist(i,k) = dhat_shift_dep_multi(...
                                    reshape(cent(k,:,:),m,d),...
                                    reshape(X(i,:,:),m,d),...
                                    shift...
                                    );
            end
        end
        [~, mem] = min(Dist,[],2);
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
