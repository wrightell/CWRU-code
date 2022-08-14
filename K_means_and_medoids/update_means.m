function means = update_means(k,data,assignment)
    dim = size(data);
    means = zeros(dim(1),k);
    for j = 1:k
        cluster = data(:,assignment == j);
        means(:,j) = mean(cluster,2);
    end
end