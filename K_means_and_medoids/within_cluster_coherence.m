function coherence = within_cluster_coherence(data,assignment,means)
    k = length(means(1,:));
    coherence = zeros(1,k);
    for i = 1:k
        cluster = data(:,assignment==i);
        coherence(1,i) = sum(vecnorm(cluster - means(:,i)))^2;
    end
end