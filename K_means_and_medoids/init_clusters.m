function [clusters,assignment,best_tightness] = init_clusters(data,k,n)
    p = length(data);
    best = [];
    best_tightness = Inf;
    for i = 1:n
        clusters = data(:,randsample(p,k));
        labels = update_assignment(clusters,data);
        tightness = sum(within_cluster_coherence(data,labels,clusters));
        if tightness < best_tightness
            best_tightness = tightness;
            best = clusters;
            assignment = labels;
        end
    end
end