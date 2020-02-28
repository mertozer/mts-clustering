function [mem, centers] = dipm(X, pval, splitRat, shift, clustering_func)

if isequal(clustering_func,@multidim_KSC)
    distFunc = @dhat_shift_dep_multi;
    averagingFunc = @ksc_centroid;
    ident = 'ksc';
else
    if isequal(clustering_func,@multidim_kShape)
        distFunc = @SBD_dep_multi;
        averagingFunc = @kshape_centroid;
        ident = 'kshape';
    else
        if isequal(clustering_func,@multidim_kDBA)
            distFunc = @cDTW_dep_multi;
            averagingFunc = @dba_centroid;
            ident = 'kdba';
        else
            if isequal(clustering_func,@multidim_kMeans)
                distFunc = @ED_dep_multi;
                averagingFunc = @kmeans_centroid;
                ident = 'kmeans';
            else
                display('please enter a valid clustering function: [multidim_KSC,multidim_kShape,multidim_kDBA,multidim_kMeans]');
                return
            end
        end
    end
end


addpath('hartigan_dip/.')
smallest_cluster        = 2;  % number of object that a cluster may have so that it doesn't split further

if ( nargin < 5 )
    error('Syntax : [groups, centers] = dipm(data, pval, splitRat, shift, @clustering func])');
    return;
end

[n,m,d]    = size(X);
k        = 2;
tX_ids   = 1:n; 
ready    = 0;
aC  = 0;
tClus  = {};
clus   = {1:n};
centers  = [];

distFile = strcat('distances/',ident,'/pairwise_n_',num2str(n),'_m_',num2str(m),'_d_',num2str(d),'_shift_',num2str(shift),'.mat');

if exist(distFile, 'file') == 2
    projected = load(distFile);
    projected = projected.projected;
else
    %% pairwise distances
    projected = zeros(size(X,1));
    for i = 1:size(X,1)
        i
        for j = i:size(X,1)
            projected(i,j) = distFunc(reshape(X(i,:,:),m,d),reshape(X(j,:,:),m,d),shift);
            projected(j,i) = projected(i,j);
        end
    end
    save(distFile,'projected');
end

hartigFile = strcat('distances/',ident,'/hartigan_n_',num2str(n),'_m_',num2str(m),'_d_',num2str(d),'_shift_',num2str(shift),'.mat');
if exist(hartigFile, 'file') == 2
    dat = load(hartigFile);
    dips = dat.dips;
    ps = dat.ps;
else
    ps = zeros(size(X,1),1);
    dips = zeros(size(X,1),1);
    for i = 1:size(X,1)
        i
        [dip,p] = HartigansDipSignifTest(projected(i,:),500);
        ps(i) = p;
        dips(i) = dip;
    end
    save(hartigFile,'dips','ps');
end

score = length(find(ps<=pval))/length(ps);
display('Initial Checks Done...');
%if ( score < splitRat )
%    cent_init = averagingFunc(ones(size(X,1),1),X,1,(sum(X,1)/n),shift);
%    centers(1,:,:)  = cent_init;
%    mem             = ones(n,1);
%    return;
%end

%% initialize two centers based on best splitter (people's front of judea)
[~,splitViewer] = max(dips);

err = intmax;
for kmi = 1:10
    [ncm_, ~, err_] = kmeans(projected(splitViewer,:)',2,'Start',[min(projected(splitViewer,projected(splitViewer,:)~=0));max(projected(splitViewer,projected(splitViewer,:)~=0))]);
    err_ = sum(err_);
    if err_ < err
        ncm = ncm_;
        err = err_;
    end
end

%ncm = kmeans(projected(splitViewer,:)',2,'Start',[min(projected(splitViewer,:));max(projected(splitViewer,:))]);
rcenters(1,:,:) = averagingFunc(ones(nnz(ncm==1),1), X(ncm == 1,:,:),1,sum(X(ncm==1,:,:),1)/n,shift);
rcenters(2,:,:) = averagingFunc(ones(nnz(ncm==2),1), X(ncm == 2,:,:),1,sum(X(ncm==2,:,:),1)/n,shift);


while (ready == 0)
    ready    = 1;
    
    scores = [];
    taC    = 0;    
    tnaC   = 0;   

    maxscore           = 0;    % variables to keep track of the maximum ad over the classified clusters
    maxscore_rindex     = 0;
    maxscore_gindex    = 0;

    
    % perform clustering
    [mem, tcenters] = clustering_func( X(tX_ids,:,:), k, shift, rcenters);
    rcenters = [];
    rdip = {};
    rprojected = {};
    rX_ids   = [];  % running ids
    
    % store the clusters separately
    
    for ki=1:k
        tClus{ki} = tX_ids(mem == ki);
        
        % calculate the dip-test statistic
        if ( length(tClus{ki}) < smallest_cluster || length(unique(mem)) == 1 )
            scores(ki) = 0;
        else
            projected_ = projected(tClus{ki},tClus{ki});
            ps = zeros(length(tClus{ki}),1);
            dips = zeros(length(tClus{ki}),1);
            parfor ii = 1:length(tClus{ki})
                [dip,p] = HartigansDipSignifTest(projected_(ii,:),1000);
                ps(ii) = p;
                dips(ii) = dip;
            end
            scores(ki) = length(find(ps<=pval))/length(ps);
        end

        % check if this cluster is accepted and do some accounting
        if ( scores(ki) <= splitRat)
            taC = taC + 1;
            aC  = aC  + 1;
            clus{aC}         = tClus{ki};
            if length(size(X)) == 3
                centers(aC,:,:)  = tcenters(ki,:,:);
            else
                if length(size(X)) == 2
                    centers(aC,:)  = tcenters(ki,:);
                end
            end
            
        else
            ready              = 0;
            tnaC               = tnaC + 1;
            rX_ids             = [rX_ids tClus{ki}];
            rdip{tnaC}         = dips;
            rprojected{tnaC}   = projected_;
            rcenters(tnaC,:,:) = tcenters(ki,:,:);
            if ( scores(ki) > maxscore || maxscore == 0 )
                maxscore           = scores(ki);
                maxscore_rindex    = tnaC; %turn index of highest score cluster among non accepted
                maxscore_gindex    = ki; % general index of highest score cluster
            end
        end
    end
    display(strcat('# of accepted clus this turn: ',num2str(taC),', # of non-accepted clus this turn: ',num2str(tnaC),...
                        ', # of accepted clus overall: ',num2str(aC)));
    if ( ready == 0 )
        k  = k + 1 - taC; % increase the k, take out the accepted clusters from the data
        tX_ids = rX_ids;

        % split up the highest multimodal cluster with its best splitter
        [~,splitViewer] = max(rdip{maxscore_rindex});
        view = reshape(rprojected{maxscore_rindex}(splitViewer,:),1,size(rprojected{maxscore_rindex},2));
        err = intmax;
        for kmi = 1:10
            [ncm_, ~, err_] = kmeans(view',2,'Start',[min(view(view~=0));max(view(view~=0))]);
            err_ = sum(err_);
            if err_ < err
                ncm = ncm_;
                err = err_;
            end
        end
        
        sClus = tClus{maxscore_gindex};
        rcenters(k,:,:) = averagingFunc(ones(nnz(ncm==2),1), X(sClus(ncm == 2),:,:),1,rcenters(maxscore_rindex,:,:),shift);
        rcenters(maxscore_rindex,:,:) = averagingFunc(ones(nnz(ncm==1),1), X(sClus(ncm == 1),:,:),1,rcenters(maxscore_rindex,:,:),shift);
    end
end

% create the Idx matrix for the resulting partition
mem = zeros(n,1);
for ki=1:length(clus)
    mem(clus{ki}) = ki;
end




